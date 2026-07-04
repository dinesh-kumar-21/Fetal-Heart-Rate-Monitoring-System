/*
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║        FETAL HEART RATE ESTIMATION — ESP32  v2.2                ║
 * ║        MicroSD (SPI) + OLED SSD1306 (I2C)                       ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  MicroSD → ESP32 :  MISO=19  MOSI=23  SCK=18  CS=5             ║
 * ║  OLED SSD1306     :  SDA=21  SCL=22                             ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Required Libraries (install via Library Manager):              ║
 * ║  • Adafruit SSD1306        (Adafruit)                           ║
 * ║  • Adafruit GFX Library    (Adafruit)                           ║
 * ║  • SD (built-in ESP32 core)                                     ║
 * ║  • SPI (built-in)                                               ║
 * ║  • Wire (built-in)                                              ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  FIX v2.1: Stack overflow resolved.                             ║
 * ║  All large arrays moved to global scope (DRAM).                 ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  FIX v2.2: Two bugs fixed:                                      ║
 * ║  1) WAV header parser — now properly walks all RIFF chunks      ║
 * ║     (handles variable-length fmt, LIST/INFO, junk chunks, etc.) ║
 * ║     Root cause: fixed 44-byte read assumed fmt chunk = 16 bytes ║
 * ║     and data chunk always at offset 36, which is wrong for      ║
 * ║     most real-world WAV files → "Invalid WAV header" error.     ║
 * ║  2) Garbled serial input — Serial RX buffer is now flushed      ║
 * ║     before every readline so leftover CR/LF/garbage bytes from  ║
 * ║     the previous loop iteration cannot corrupt the filename.    ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  DATASET: IIScFHSDB (PhysioNet)                                 ║
 * ║  • 60 subjects, WAV mono 16-bit, Fs = 2000 Hz, ~8 min each     ║
 * ║  • Files: subject_01.wav … subject_60.wav                      ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// ── PIN DEFINITIONS ──────────────────────────────────────────────────
#define SD_CS    5
#define SD_MOSI  23
#define SD_MISO  19
#define SD_SCK   18

#define OLED_SDA  21
#define OLED_SCL  22
#define OLED_W    128
#define OLED_H    64
#define OLED_ADDR 0x3C

// ── ALGORITHM PARAMETERS ─────────────────────────────────────────────
#define CHUNK_SAMPLES  1024
#define ENV_SMOOTH_MS  80

#define MIN_FHR_BPM    110
#define MAX_FHR_BPM    160
#define MIN_MAT_BPM    60
#define MAX_MAT_BPM    100

#define MAX_PEAKS      400
#define MA_BUF_SIZE    200

// ── IIR BIQUAD COEFFICIENTS ──────────────────────────────────────────
// 4th-order Butterworth bandpass 20–100 Hz @ Fs = 2000 Hz
static const float bpSOS[2][5] = {
  { 0.12274f,  0.0f, -0.12274f, -1.92445f,  0.92935f },
  { 0.10952f,  0.0f, -0.10952f, -1.68547f,  0.75327f }
};

// ── BIQUAD STATE ─────────────────────────────────────────────────────
struct BiquadState { float x1, x2, y1, y2; };
BiquadState bqState[2];

float biquad(float x, int s) {
  float y = bpSOS[s][0] * x
          + bpSOS[s][1] * bqState[s].x1
          + bpSOS[s][2] * bqState[s].x2
          - bpSOS[s][3] * bqState[s].y1
          - bpSOS[s][4] * bqState[s].y2;
  bqState[s].x2 = bqState[s].x1;  bqState[s].x1 = x;
  bqState[s].y2 = bqState[s].y1;  bqState[s].y1 = y;
  return y;
}

inline float filterSample(float x) {
  return biquad(biquad(x, 0), 1);
}

// ═══════════════════════════════════════════════════════════════
// All large arrays are global to avoid stack overflow
// ═══════════════════════════════════════════════════════════════
static uint8_t  raw8 [CHUNK_SAMPLES];
static int16_t  raw16[CHUNK_SAMPLES];
static uint32_t peakBuf[MAX_PEAKS];
static float    bpmWork[MAX_PEAKS];
static float    maBuf[MA_BUF_SIZE];

Adafruit_SSD1306 oled(OLED_W, OLED_H, &Wire, -1);

uint32_t peakCount   = 0;
float    poolSD      = 0.0f;
String   dominanceFlag = "---";

uint32_t wavFs       = 2000;
uint8_t  wavBits     = 16;
uint32_t wavDataSize = 0;

// ── MOVING AVERAGE ────────────────────────────────────────────────────
int   maHead = 0;
float maSum  = 0.0f;
int   maFill = 0;
int   maWin  = 160;

void maInit(int winSamples) {
  maWin  = constrain(winSamples, 1, MA_BUF_SIZE);
  maHead = 0; maSum = 0.0f; maFill = 0;
  memset(maBuf, 0, maWin * sizeof(float));
}

float maPush(float v) {
  if (maFill == maWin) maSum -= maBuf[maHead];
  else                 maFill++;
  maBuf[maHead] = v;
  maHead = (maHead + 1) % maWin;
  maSum += v;
  return maSum / maFill;
}

// ── PEAK DETECTOR ─────────────────────────────────────────────────────
float    pkPrev = 0.0f, pkPrev2 = 0.0f;
bool     pkRising   = false;
uint32_t pkLockout  = 0;
uint32_t pkSampleN  = 0;
float    dynThresh  = 0.35f;

bool peakDetect(float env) {
  bool found = false;
  if (env > pkPrev && pkPrev >= pkPrev2) pkRising = true;

  if (pkRising && env < pkPrev) {
    uint32_t refract = (uint32_t)(0.32f * wavFs);
    if (pkSampleN > pkLockout && pkPrev > dynThresh) {
      found     = true;
      pkLockout = pkSampleN + refract;
    }
    pkRising = false;
  }
  dynThresh = dynThresh * 0.998f + pkPrev * 0.002f;
  dynThresh = constrain(dynThresh, 0.15f, 0.75f);
  pkPrev2 = pkPrev;
  pkPrev  = env;
  return found;
}

// ── FHR COMPUTATION ───────────────────────────────────────────────────
bool computeFHR(float &fhrMean, float &fhrSD) {
  if (peakCount < 4) return false;

  uint32_t valid = 0;
  for (uint32_t i = 1; i < peakCount && valid < MAX_PEAKS; i++) {
    float dt  = (float)(peakBuf[i] - peakBuf[i-1]) / (float)wavFs;
    if (dt < 0.001f) continue;
    float bpm = 60.0f / dt;
    if (bpm >= MIN_FHR_BPM && bpm <= MAX_FHR_BPM)
      bpmWork[valid++] = bpm;
  }
  if (valid < 3) return false;

  for (uint32_t i = 0; i < valid - 1; i++)
    for (uint32_t j = i + 1; j < valid; j++)
      if (bpmWork[j] < bpmWork[i]) {
        float t = bpmWork[i]; bpmWork[i] = bpmWork[j]; bpmWork[j] = t;
      }

  float med = bpmWork[valid / 2];

  float sum = 0.0f, sum2 = 0.0f; uint32_t cnt = 0;
  for (uint32_t i = 0; i < valid; i++) {
    if (fabsf(bpmWork[i] - med) <= 0.15f * med) {
      sum  += bpmWork[i];
      sum2 += bpmWork[i] * bpmWork[i];
      cnt++;
    }
  }
  if (cnt < 2) return false;

  fhrMean = sum / cnt;
  fhrSD   = sqrtf(fmaxf(sum2 / cnt - fhrMean * fhrMean, 0.0f));
  return true;
}

// ── DOMINANCE FLAG ────────────────────────────────────────────────────
String dominance(float ratio) {
  if (ratio >= 0.85f) return "FETAL-DOM";
  if (ratio >= 0.60f) return "MIXED";
  return "MATERNAL";
}

// ── OLED HELPERS ──────────────────────────────────────────────────────
void oledSplash() {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(4,  2); oled.println(F("FHR Estimator v2.2"));
  oled.drawLine(0, 12, 127, 12, SSD1306_WHITE);
  oled.setCursor(4, 17); oled.println(F("IIScFHSDB ready"));
  oled.setCursor(4, 29); oled.println(F("ESP32 + MicroSD"));
  oled.setCursor(4, 41); oled.println(F("OLED SSD1306"));
  oled.setCursor(4, 54); oled.println(F("Serial @ 115200 baud"));
  oled.display();
}

void oledStatus(const char *l1, const char *l2 = "", const char *l3 = "") {
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0,  0); oled.println(l1);
  oled.setCursor(0, 18); oled.println(l2);
  oled.setCursor(0, 36); oled.println(l3);
  oled.display();
}

void oledProgress(float pct, float liveFHR) {
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0); oled.println(F("Processing..."));

  int barW = (int)(pct * 124.0f);
  oled.drawRect(2, 14, 124, 10, SSD1306_WHITE);
  if (barW > 0) oled.fillRect(2, 14, barW, 10, SSD1306_WHITE);

  oled.setCursor(48, 28);
  oled.print((int)(pct * 100.0f));
  oled.print(F(" %"));

  if (liveFHR > 0.0f) {
    oled.setCursor(0, 42);
    oled.print(F("Live: "));
    oled.print(liveFHR, 1);
    oled.print(F(" BPM"));
  }
  oled.display();
}

void oledResult(float fhr, float sd, const String &flag, float gt, bool hasGT) {
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(16, 0); oled.println(F("=== FHR RESULT ==="));
  oled.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  oled.setTextSize(2);
  oled.setCursor(4, 14);
  oled.print(fhr, 1);
  oled.print(F(" BPM"));

  oled.setTextSize(1);
  oled.setCursor(0, 36);
  oled.print(F("SD:")); oled.print(sd, 1);
  oled.print(F("  ")); oled.print(flag);

  oled.setCursor(0, 50);
  if (hasGT && gt > 0.0f) {
    oled.print(F("MAE:"));
    oled.print(fabsf(fhr - gt), 1);
    oled.print(F(" GT:"));
    oled.print(gt, 0);
  } else {
    oled.print(F("No ground truth"));
  }
  oled.display();
}

// ══════════════════════════════════════════════════════════════════════
// WAV HEADER PARSER  —  v2.2 FIX
//
// OLD BUG: read exactly 44 bytes and assumed the layout was:
//   [RIFF 12B][fmt  8B+16B data][data 8B header] = 44 bytes
// This breaks for ANY file where:
//   • the fmt chunk has extra bytes (18-byte or 40-byte extended PCM)
//   • there are extra chunks before "data" (LIST, INFO, junk, bext …)
//
// NEW FIX: read the 12-byte RIFF/WAVE preamble, then walk every
// sub-chunk properly (4-byte ID + 4-byte size + skip data) until
// we have found both "fmt " and "data".  Odd-sized chunks are
// padded to even boundaries per the RIFF spec, so we honour that.
// ══════════════════════════════════════════════════════════════════════
bool parseWAVHeader(File &f) {

  // ── 1. RIFF/WAVE preamble ──────────────────────────────────────────
  uint8_t preamble[12];
  if (f.read(preamble, 12) != 12)               return false;
  if (preamble[0]!='R' || preamble[1]!='I' ||
      preamble[2]!='F' || preamble[3]!='F')     return false;
  if (preamble[8]!='W' || preamble[9]!='A' ||
      preamble[10]!='V'|| preamble[11]!='E')    return false;

  // ── 2. Walk sub-chunks ────────────────────────────────────────────
  wavFs = 0; wavBits = 0; wavDataSize = 0;
  uint16_t nCh = 0;
  bool fmtFound  = false;
  bool dataFound = false;

  // Safety limit: scan at most 32 chunks (more than enough for any
  // real WAV file; prevents infinite loop on corrupt files).
  for (int iter = 0; iter < 32 && !dataFound; iter++) {

    uint8_t chunkHdr[8];
    if (f.read(chunkHdr, 8) != 8) break;   // EOF or read error

    uint32_t chunkSize =
        (uint32_t)chunkHdr[4]        |
        ((uint32_t)chunkHdr[5] << 8) |
        ((uint32_t)chunkHdr[6] << 16)|
        ((uint32_t)chunkHdr[7] << 24);

    // ── "fmt " chunk ────────────────────────────────────────────────
    if (chunkHdr[0]=='f' && chunkHdr[1]=='m' &&
        chunkHdr[2]=='t' && chunkHdr[3]==' ') {

      // fmt data is at least 16 bytes; may be 18 (PCM extended) or
      // 40 (WAVE_FORMAT_EXTENSIBLE).  Read up to 40, skip the rest.
      uint8_t  fmt[40];
      uint32_t toRead = (chunkSize < 40) ? chunkSize : 40;
      if ((uint32_t)f.read(fmt, toRead) != toRead) return false;

      // Skip any bytes beyond what we read (rare: chunk > 40 bytes)
      if (chunkSize > toRead) {
        uint32_t skip = chunkSize - toRead;
        if (!f.seek(f.position() + skip)) return false;
      }

      // fmt chunk data layout (offsets from start of chunk DATA):
      //  0- 1  AudioFormat   (1 = PCM)
      //  2- 3  NumChannels
      //  4- 7  SampleRate
      //  8-11  ByteRate
      // 12-13  BlockAlign
      // 14-15  BitsPerSample
      nCh    = (uint16_t)fmt[2] | ((uint16_t)fmt[3] << 8);
      wavFs  = (uint32_t)fmt[4] | ((uint32_t)fmt[5] << 8)
             | ((uint32_t)fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
      wavBits = (uint16_t)fmt[14] | ((uint16_t)fmt[15] << 8);
      fmtFound = true;

      // RIFF spec: chunks are word-aligned (even size).
      // If chunk size is odd, one padding byte follows.
      if (chunkSize & 1) f.seek(f.position() + 1);
    }
    // ── "data" chunk ────────────────────────────────────────────────
    else if (chunkHdr[0]=='d' && chunkHdr[1]=='a' &&
             chunkHdr[2]=='t' && chunkHdr[3]=='a') {

      wavDataSize = chunkSize;
      dataFound   = true;
      // File pointer now sits at the first audio sample — do NOT seek.
    }
    // ── Any other chunk (LIST, INFO, junk, bext, smpl …) ───────────
    else {
      // Skip the entire chunk body (+ 1 pad byte if odd-sized).
      uint32_t skip = chunkSize + (chunkSize & 1);
      if (!f.seek(f.position() + skip)) break;
    }
  }

  if (!fmtFound || !dataFound) return false;

  Serial.printf("  WAV   Fs=%u Hz | %u-bit | %u ch | %.1f s\n",
    wavFs, wavBits, nCh,
    (float)wavDataSize / (float)(wavBits / 8) / (float)wavFs);

  if (nCh != 1)
    Serial.println(F("  WARNING: Multi-channel — convert to mono first."));

  return true;
}

// ── MAIN PROCESSING FUNCTION ──────────────────────────────────────────
float processWAV(File &f) {

  uint32_t bytesPerSample = (wavBits == 16) ? 2 : 1;
  uint32_t totalSamples   = wavDataSize / bytesPerSample;

  Serial.printf("  Samples : %u  (%.1f s)\n",
                totalSamples, (float)totalSamples / wavFs);

  maInit((int)(ENV_SMOOTH_MS * 0.001f * wavFs));
  memset(bqState, 0, sizeof(bqState));
  memset(peakBuf, 0, sizeof(peakBuf));
  peakCount   = 0;
  pkSampleN   = 0;
  pkPrev = pkPrev2 = 0.0f;
  pkRising    = false;
  pkLockout   = 0;
  dynThresh   = 0.35f;

  float    runMax       = 1e-6f;
  float    liveFHR      = 0.0f;
  uint32_t samplesRead  = 0;
  uint32_t lastOLED     = 0;

  while (samplesRead < totalSamples) {
    uint32_t toRead      = min((uint32_t)CHUNK_SAMPLES, totalSamples - samplesRead);
    uint32_t bytesToRead = toRead * bytesPerSample;

    int got;
    if (wavBits == 16) {
      got = f.read((uint8_t*)raw16, bytesToRead) / 2;
    } else {
      got = f.read(raw8, bytesToRead);
    }
    if (got <= 0) break;

    for (int i = 0; i < got; i++) {

      float x;
      if (wavBits == 16) x = (float)raw16[i] / 32768.0f;
      else               x = (float)(raw8[i] - 128) / 128.0f;

      float filtered = filterSample(x);

      float x2 = filtered * filtered;
      float se = (x2 < 1e-10f) ? 0.0f : -(x2 * logf(x2 + 1e-10f));

      float absVal = fabsf(se);
      if (absVal > runMax) runMax = absVal;
      else                 runMax *= 0.99999f;
      float env = maPush(absVal / runMax);

      if (peakDetect(env)) {
        if (peakCount < MAX_PEAKS) {
          peakBuf[peakCount++] = pkSampleN;
        } else {
          memmove(peakBuf, peakBuf + 1, (MAX_PEAKS - 1) * sizeof(uint32_t));
          peakBuf[MAX_PEAKS - 1] = pkSampleN;
        }
      }
      pkSampleN++;
    }

    samplesRead += got;

    if (peakCount >= 6) {
      float lm, ls;
      if (computeFHR(lm, ls)) liveFHR = lm;
    }

    uint32_t now = millis();
    if (now - lastOLED > 500) {
      oledProgress((float)samplesRead / (float)totalSamples, liveFHR);
      lastOLED = now;
    }
  }

  float fhrMean = 0.0f, fhrSD = 0.0f;
  bool  ok = computeFHR(fhrMean, fhrSD);

  float fetalCnt = 0.0f, matCnt = 0.0f;
  for (uint32_t i = 1; i < peakCount; i++) {
    float dt  = (float)(peakBuf[i] - peakBuf[i-1]) / wavFs;
    if (dt < 0.001f) continue;
    float bpm = 60.0f / dt;
    if (bpm >= MIN_FHR_BPM && bpm <= MAX_FHR_BPM) fetalCnt++;
    if (bpm >= MIN_MAT_BPM  && bpm <= MAX_MAT_BPM)  matCnt++;
  }
  dominanceFlag = dominance(fetalCnt / fmaxf(matCnt, 1.0f));
  poolSD        = fhrSD;

  return ok ? fhrMean : -1.0f;
}

// ══════════════════════════════════════════════════════════════════════
// SERIAL HELPER  —  v2.2 FIX
//
// OLD BUG: loop() called Serial.readStringUntil('\n') without first
// flushing the RX buffer.  At the end of the previous iteration the
// 15-second delay() expires while the terminal may have sent CR/LF
// or auto-repeat characters; those bytes sit in the 256-byte HW FIFO
// and are read as the "filename" on the next iteration, producing the
// garbled     … strings seen in the log.
//
// NEW FIX: serialReadLine() drains every byte currently in the buffer,
// waits until the user actually presses Enter, then returns the clean
// trimmed string.
// ══════════════════════════════════════════════════════════════════════
String serialReadLine(const char *prompt) {
  Serial.print(prompt);

  // Drain any stale bytes that arrived while we were not reading
  // (e.g. CR/LF from the previous readline, terminal keep-alive, etc.)
  delay(20);                          // let UART finish receiving last char
  while (Serial.available()) Serial.read();

  // Now block until the user types something and presses Enter
  while (!Serial.available()) delay(50);
  String s = Serial.readStringUntil('\n');
  s.trim();
  Serial.println(s);                  // echo so the log shows what was typed
  return s;
}

// ══════════════════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(600);

  Wire.begin(OLED_SDA, OLED_SCL);
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("ERROR: OLED not found. Check GPIO21/22."));
    while (true) delay(1000);
  }
  oled.setTextWrap(false);
  oledSplash();

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS)) {
    Serial.println(F("ERROR: SD card not found."));
    oledStatus("SD CARD ERROR", "Check wiring:", "GPIO5/18/19/23");
    while (true) delay(1000);
  }
  Serial.println(F("SD card OK."));

  Serial.println();
  Serial.println(F("╔════════════════════════════════════════════════╗"));
  Serial.println(F("║   FETAL HEART RATE ESTIMATION — ESP32  v2.2   ║"));
  Serial.println(F("║   Dataset : IIScFHSDB (PhysioNet)             ║"));
  Serial.println(F("║   Fix     : WAV parser + serial flush          ║"));
  Serial.println(F("╚════════════════════════════════════════════════╝"));
  Serial.println();

  Serial.printf("  Free heap : %u bytes\n\n", ESP.getFreeHeap());
}

// ══════════════════════════════════════════════════════════════════════
// LOOP
// ══════════════════════════════════════════════════════════════════════
void loop() {

  Serial.println(F("────────────────────────────────────────────────"));
  Serial.println(F("  IIScFHSDB files: subject_01.wav … subject_60.wav"));

  oledStatus("Ready", "Enter filename", "in Serial Monitor");

  // ── FIX: serialReadLine() flushes stale bytes before blocking ──────
  String filename = serialReadLine("  Enter filename (e.g. subject_04.wav): ");

  if (filename.length() == 0) {
    Serial.println(F("  No filename entered. Try again."));
    return;
  }
  if (!filename.startsWith("/")) filename = "/" + filename;

  String gtStr = serialReadLine("  Enter ground truth FHR in BPM (0 = skip): ");
  float groundTruth = gtStr.toFloat();
  bool  hasGT       = (groundTruth >= 50.0f && groundTruth <= 220.0f);

  if (hasGT) Serial.printf("  Ground truth : %.0f BPM\n\n", groundTruth);
  else        Serial.println(F("  Ground truth : not set\n"));

  oledStatus("Opening...", filename.c_str(), "");
  File f = SD.open(filename);
  if (!f) {
    Serial.printf("  ERROR: Cannot open %s\n", filename.c_str());
    oledStatus("FILE NOT FOUND", filename.c_str(), "Check SD card");
    delay(3000); return;
  }
  Serial.printf("  File size : %u bytes\n", (unsigned)f.size());

  // ── FIX: parseWAVHeader now walks chunks properly ──────────────────
  if (!parseWAVHeader(f)) {
    Serial.println(F("  ERROR: Invalid WAV header."));
    Serial.println(F("  (Checked RIFF/WAVE magic + walked all sub-chunks)"));
    oledStatus("INVALID WAV", "Need PCM mono", "16-bit 2000 Hz");
    f.close(); delay(3000); return;
  }

  if (wavFs != 2000)
    Serial.printf("  NOTE: Fs=%u Hz — coefficients optimised for 2000 Hz.\n", wavFs);

  oledStatus("Processing...", "Please wait", "~1-2 minutes");
  delay(300);
  Serial.println(F("\n  Processing..."));

  uint32_t t0      = millis();
  float    result  = processWAV(f);
  float    elapsed = (millis() - t0) / 1000.0f;
  f.close();

  Serial.println();
  Serial.println(F("── RESULTS ──────────────────────────────────────────"));
  if (result > 0.0f) {
    Serial.printf("  Estimated FHR  : %.2f BPM\n",  result);
    Serial.printf("  Pool SD        : %.2f BPM\n",  poolSD);
    Serial.printf("  Dominance      : %s\n",         dominanceFlag.c_str());
    Serial.printf("  Peaks detected : %u\n",         peakCount);
    if (hasGT) {
      float mae = fabsf(result - groundTruth);
      Serial.printf("  Ground Truth   : %.0f BPM\n", groundTruth);
      Serial.printf("  MAE            : %.2f BPM\n", mae);
      Serial.printf("  Accuracy       : %s\n",
        mae <= 5.0f  ? "EXCELLENT (<=5 BPM)"  :
        mae <= 10.0f ? "GOOD (<=10 BPM)"       : "CHECK SIGNAL");
    }
  } else {
    Serial.println(F("  FAILED — not enough valid peaks detected."));
    Serial.println(F("  Tips: ensure file is 16-bit PCM mono 2000 Hz"));
  }
  Serial.printf("  Processing time: %.1f s\n",   elapsed);
  Serial.printf("  Free heap after: %u bytes\n", ESP.getFreeHeap());
  Serial.println(F("─────────────────────────────────────────────────────\n"));

  if (result > 0.0f)
    oledResult(result, poolSD, dominanceFlag, groundTruth, hasGT);
  else
    oledStatus("FAILED", "Not enough peaks", "Check Serial");

  delay(15000);
}