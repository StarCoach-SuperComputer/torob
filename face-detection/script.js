// Global state
let recognition = null;
let isRecording = false;
let currentEmotion = "neutral";
let currentAge = null;
let currentGender = null;
let activeSessionId = null;

// Identity detection state
const STORAGE_KEY = "torob_face_identities";
const VISION_CACHE_KEY = "torob_vision_profiles_cache";
const ONE_DAY_MS = 24 * 60 * 60 * 1000; // 1 day in milliseconds (86,400,000)

let storedIdentities = [];
let faceMatcher = null;
let currentIdentity = null; // { id, name, similarity }
let latestDescriptor = null;
let unknownFaceFrameCount = 0;
let autoAssignNewFaces = true;

// Backend API URL
const BACKEND_URL = "http://localhost:8000";

/**
 * Storage & Identity Management
 */
function loadStoredIdentities() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    storedIdentities = raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error("Error reading localStorage:", e);
    storedIdentities = [];
  }
  rebuildFaceMatcher();
  renderIdentitiesList();
}

function saveStoredIdentities() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(storedIdentities));
  } catch (e) {
    console.error("Error writing to localStorage:", e);
  }
  rebuildFaceMatcher();
  renderIdentitiesList();
}

function rebuildFaceMatcher() {
  if (!storedIdentities || storedIdentities.length === 0) {
    faceMatcher = null;
    return;
  }

  const labeledDescriptors = storedIdentities
    .filter(item => item.descriptors && item.descriptors.length > 0)
    .map(item => {
      const floatDescriptors = item.descriptors.map(d => new Float32Array(d));
      return new faceapi.LabeledFaceDescriptors(item.name || item.id, floatDescriptors);
    });

  if (labeledDescriptors.length > 0) {
    // 0.55 distance threshold: lower = stricter, higher = more tolerant
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.55);
  } else {
    faceMatcher = null;
  }
}

function getNextUserLabel() {
  const existingNumbers = storedIdentities.map(u => {
    const match = u.name.match(/کاربر\s*(\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  });
  const maxNum = existingNumbers.length > 0 ? Math.max(...existingNumbers) : 0;
  return `کاربر ${maxNum + 1}`;
}

function registerFace(customName = null) {
  if (!latestDescriptor) {
    alert("هیچ چهره‌ای در تصویر برای ثبت پیدا نشد!");
    return null;
  }

  const name = customName && customName.trim() ? customName.trim() : getNextUserLabel();
  const existing = storedIdentities.find(u => u.name === name);

  if (existing) {
    // Add additional sample (keep up to 5 samples per person to boost accuracy across angles)
    existing.descriptors.push(Array.from(latestDescriptor));
    if (existing.descriptors.length > 5) {
      existing.descriptors.shift();
    }
  } else {
    storedIdentities.push({
      id: `user_${Date.now()}`,
      name: name,
      createdAt: new Date().toISOString(),
      descriptors: [Array.from(latestDescriptor)]
    });
  }

  saveStoredIdentities();
  return name;
}

function deleteIdentity(id) {
  storedIdentities = storedIdentities.filter(u => u.id !== id);
  saveStoredIdentities();
  deleteCachedVisionProfile(id);
}

function clearAllIdentities() {
  if (confirm("آیا از حذف تمام چهره‌های ذخیره شده اطمینان دارید؟")) {
    storedIdentities = [];
    saveStoredIdentities();
    clearAllVisionCache();
  }
}

/**
 * Client-Side Vision Profile Cache (1-day TTL for Token Cost Optimization)
 */
function getCachedVisionProfile(userKey) {
  try {
    const raw = localStorage.getItem(VISION_CACHE_KEY);
    if (!raw) return null;
    const cache = JSON.parse(raw);
    const entry = cache[userKey];
    if (!entry) return null;

    const age = Date.now() - entry.timestamp;
    if (age > ONE_DAY_MS) {
      console.log(`[Vision Cache] Profile for ${userKey} expired (older than 1 day: ${(age / 3600000).toFixed(1)}h).`);
      delete cache[userKey];
      localStorage.setItem(VISION_CACHE_KEY, JSON.stringify(cache));
      return null;
    }

    console.log(`[Vision Cache] Valid profile for ${userKey} found (age: ${(age / 60000).toFixed(0)}m). Skipping image upload.`);
    return entry;
  } catch (e) {
    console.error("Error reading vision cache:", e);
    return null;
  }
}

function setCachedVisionProfile(userKey, profileData) {
  try {
    const raw = localStorage.getItem(VISION_CACHE_KEY);
    const cache = raw ? JSON.parse(raw) : {};
    cache[userKey] = {
      timestamp: Date.now(),
      profile: profileData,
      savedAt: new Date().toISOString()
    };
    localStorage.setItem(VISION_CACHE_KEY, JSON.stringify(cache));
    console.log(`[Vision Cache] Stored profile for ${userKey} at ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    console.error("Error storing vision cache:", e);
  }
}

function deleteCachedVisionProfile(userKey) {
  try {
    const raw = localStorage.getItem(VISION_CACHE_KEY);
    if (!raw) return;
    const cache = JSON.parse(raw);
    delete cache[userKey];
    localStorage.setItem(VISION_CACHE_KEY, JSON.stringify(cache));
  } catch (e) {
    console.error("Error deleting vision cache entry:", e);
  }
}

function clearAllVisionCache() {
  try {
    localStorage.removeItem(VISION_CACHE_KEY);
  } catch (e) {
    console.error("Error clearing vision cache:", e);
  }
}

function renderIdentitiesList() {
  const listContainer = document.getElementById('identitiesList');
  const countSpan = document.getElementById('storedCount');
  if (countSpan) countSpan.innerText = storedIdentities.length;

  if (!listContainer) return;

  if (storedIdentities.length === 0) {
    listContainer.innerHTML = '<span class="empty-note">هنوز هیچ چهره‌ای ذخیره نشده است.</span>';
    return;
  }

  listContainer.innerHTML = storedIdentities.map(user => `
    <div class="identity-chip" data-id="${user.id}">
      <span class="chip-name">👤 ${user.name}</span>
      <span class="chip-samples">(${user.descriptors.length} نمونه)</span>
      <span class="chip-delete" onclick="window.onDeleteIdentity('${user.id}')" title="حذف">&times;</span>
    </div>
  `).join('');
}

// Expose delete handler for inline onclick
window.onDeleteIdentity = (id) => {
  deleteIdentity(id);
};

function updateIdentityUI(identity, noFace = false) {
  const badge = document.getElementById('currentIdentityBadge');
  if (!badge) return;

  if (noFace) {
    badge.className = "identity-badge none";
    badge.innerText = "در انتظار چهره...";
  } else if (identity) {
    badge.className = "identity-badge recognized";
    badge.innerText = `👤 ${identity.name} (تطابق ${identity.similarity}٪)`;
  } else {
    badge.className = "identity-badge unknown";
    if (autoAssignNewFaces && unknownFaceFrameCount > 0) {
      badge.innerText = `❓ ناشناس (در حال اختصاص شناسه... ${unknownFaceFrameCount}/5)`;
    } else {
      badge.innerText = "❓ شخص ناشناس";
    }
  }
}

/**
 * Load face-api.js Models
 */
async function loadModels() {
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromUri('models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('models');
    await faceapi.nets.ageGenderNet.loadFromUri('models');
    await faceapi.nets.faceExpressionNet.loadFromUri('models');

    console.log("Models loaded successfully.");
    document.getElementById('loadingIndicator').style.display = 'none';

    loadStoredIdentities();
    startVideo();
  } catch (error) {
    console.error("Error loading models:", error);
    document.getElementById('loadingIndicator').innerText = "خطا در بارگذاری مدل‌های چهره";
  }
}

function startVideo() {
  const video = document.getElementById('video');
  const warningMessage = document.getElementById('warningMessage');

  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      video.classList.remove('hidden');
      warningMessage.classList.add('hidden');
      video.play();
      detectFaces();
    })
    .catch((error) => {
      console.error("Error accessing webcam:", error);
      warningMessage.classList.remove('hidden');
    });
}

async function detectFaces() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const emotionResult = document.getElementById('emotionResult');

  video.addEventListener('loadedmetadata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  });

  const getDisplaySize = () => ({
    width: video.videoWidth,
    height: video.videoHeight,
  });

  canvas.classList.remove('hidden');

  setInterval(async () => {
    if (!video.videoWidth || !video.videoHeight) return;

    // Detect faces with landmarks, expressions, age/gender, AND face descriptors
    const detections = await faceapi.detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, getDisplaySize());

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw landmarks and expressions
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    if (detections[0]) {
      latestDescriptor = detections[0].descriptor;

      const expressions = detections[0].expressions;
      const sorted = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
      currentEmotion = sorted[0][0];
      currentAge = Math.round(detections[0].age);
      currentGender = detections[0].gender;

      const emotionFaMap = {
        happy: "خوشحال / باانرژی",
        neutral: "عادی / آرام",
        sad: "ناراحت",
        surprised: "متعجب",
        angry: "عصبانی",
        fearful: "نگران",
        disgusted: "ناراضی"
      };

      const genderFa = currentGender === "male" ? "آقا" : "خانم";
      const emotionFa = emotionFaMap[currentEmotion] || currentEmotion;

      emotionResult.innerText = `چهره: ${genderFa}، حدود ${currentAge} سال | حالت چهره: ${emotionFa}`;

      // Identity Matching for primary face
      let matchedLabel = null;
      let matchSimilarity = 0;

      if (faceMatcher && detections[0].descriptor) {
        const bestMatch = faceMatcher.findBestMatch(detections[0].descriptor);
        if (bestMatch.label !== 'unknown') {
          matchedLabel = bestMatch.label;
          matchSimilarity = Math.max(0, Math.min(100, Math.round((1 - bestMatch.distance) * 100)));
        }
      }

      if (matchedLabel) {
        unknownFaceFrameCount = 0;
        const matchedUser = storedIdentities.find(u => u.name === matchedLabel);
        currentIdentity = {
          id: matchedUser ? matchedUser.id : matchedLabel,
          name: matchedLabel,
          similarity: matchSimilarity
        };
        updateIdentityUI(currentIdentity);
      } else {
        currentIdentity = null;

        // Auto-assign new identity if an unknown face stays for 5 checks (~2s)
        if (autoAssignNewFaces) {
          unknownFaceFrameCount++;
          updateIdentityUI(null);
          if (unknownFaceFrameCount >= 5) {
            const newName = registerFace();
            unknownFaceFrameCount = 0;
            console.log(`Auto-assigned new identity: ${newName}`);
          }
        } else {
          unknownFaceFrameCount = 0;
          updateIdentityUI(null);
        }
      }
    } else {
      latestDescriptor = null;
      currentIdentity = null;
      unknownFaceFrameCount = 0;
      emotionResult.innerText = "هیچ چهره‌ای در کادر دوربین شناسایی نشد.";
      updateIdentityUI(null, true);
    }

    // Draw customized bounding boxes with identity badge for each detected face
    resizedDetections.forEach((detection) => {
      const { x, y, width, height } = detection.detection.box;
      let label = "ناشناس";
      let boxColor = "#f59e0b"; // amber for unknown

      if (faceMatcher && detection.descriptor) {
        const match = faceMatcher.findBestMatch(detection.descriptor);
        if (match.label !== 'unknown') {
          const sim = Math.max(0, Math.min(100, Math.round((1 - match.distance) * 100)));
          label = `👤 ${match.label} (${sim}٪)`;
          boxColor = "#10b981"; // emerald for recognized
        }
      } else if (autoAssignNewFaces && unknownFaceFrameCount > 0) {
        label = `⏳ ثبت خودکار (${unknownFaceFrameCount}/5)...`;
      }

      // Draw bounding box
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label background badge above face
      ctx.font = "bold 13px system-ui, -apple-system, sans-serif";
      const padding = 8;
      const textWidth = ctx.measureText(label).width;
      const badgeHeight = 24;
      const badgeY = Math.max(0, y - badgeHeight - 4);

      ctx.fillStyle = boxColor;
      ctx.fillRect(x, badgeY, textWidth + (padding * 2), badgeHeight);

      // Draw label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x + padding, badgeY + 16);
    });
  }, 400);
}

/**
 * Capture frame from live video as base64 data URI
 */
function captureFrameBase64() {
  const video = document.getElementById('video');
  if (!video || !video.videoWidth) return null;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const ctx = tempCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  return tempCanvas.toDataURL('image/jpeg', 0.85);
}

/**
 * Setup Browser Web Speech API (Speech to Text)
 */
function initSpeechToText() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  const micBtn = document.getElementById('micBtn');
  const micStatusText = document.getElementById('micStatusText');
  const promptInput = document.getElementById('promptInput');
  const langSelect = document.getElementById('langSelect');

  if (!SpeechRecognition) {
    micStatusText.innerText = "مرورگر شما از Web Speech API پشتیبانی نمی‌کند (کروم پیشنهاد می‌شود)";
    micBtn.disabled = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = langSelect.value;

  langSelect.addEventListener('change', () => {
    if (recognition) {
      recognition.lang = langSelect.value;
    }
  });

  recognition.onstart = () => {
    isRecording = true;
    micBtn.classList.add('recording');
    micStatusText.innerText = "در حال شنیدن صدای شما... (صحبت کنید)";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    promptInput.value = transcript;
  };

  recognition.onerror = (event) => {
    console.error("Speech Recognition Error:", event.error);
    isRecording = false;
    micBtn.classList.remove('recording');
    micStatusText.innerText = `خطا در دریافت صدا (${event.error}) - دوباره امتحان کنید`;
  };

  recognition.onend = () => {
    isRecording = false;
    micBtn.classList.remove('recording');
    micStatusText.innerText = "کلیک کنید برای صحبت مجدد";
  };

  micBtn.addEventListener('click', () => {
    stopSpeech();
    if (isRecording) {
      recognition.stop();
    } else {
      promptInput.value = "";
      recognition.lang = langSelect.value;
      recognition.start();
    }
  });
}

/**
 * Speech Synthesis (Text to Speech - TTS) based on tts.html
 */
const synth = window.speechSynthesis;
let ttsVoices = [];
let autoTtsEnabled = true;
let lastAssistantReplyText = "";
let speechQueue = [];
let isSpeakingQueue = false;

function updateTtsStatus(msg) {
  const el = document.getElementById('ttsStatus');
  if (el) el.innerText = msg;
}

function initTextToSpeech() {
  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    console.warn("مرورگر شما از Web Speech Synthesis پشتیبانی نمی‌کند.");
    const ttsActions = document.querySelector('.tts-actions');
    if (ttsActions) ttsActions.style.display = 'none';
    updateTtsStatus("مرورگر شما از Web Speech API پشتیبانی نمی‌کند.");
    return;
  }

  loadTtsVoices();
  synth.addEventListener("voiceschanged", loadTtsVoices);

  const autoCheckbox = document.getElementById('autoTtsCheckbox');
  if (autoCheckbox) {
    autoTtsEnabled = autoCheckbox.checked;
    autoCheckbox.addEventListener('change', (e) => {
      autoTtsEnabled = e.target.checked;
      if (!autoTtsEnabled && synth.speaking) {
        stopSpeech();
      }
    });
  }

  const voiceSelect = document.getElementById('ttsVoiceSelect');
  if (voiceSelect) {
    voiceSelect.addEventListener('change', () => {
      if (synth.speaking) {
        stopSpeech();
        if (lastAssistantReplyText) {
          speakText(lastAssistantReplyText);
        }
      }
    });
  }

  const speakBtn = document.getElementById('ttsSpeakBtn');
  if (speakBtn) {
    speakBtn.addEventListener('click', () => {
      if (lastAssistantReplyText) {
        speakText(lastAssistantReplyText);
      } else {
        updateTtsStatus("هنوز پاسخی برای خواندن موجود نیست.");
      }
    });
  }

  const stopBtn = document.getElementById('ttsStopBtn');
  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      stopSpeech();
    });
  }
}

function loadTtsVoices() {
  if (!synth) return;
  ttsVoices = synth.getVoices();
  const voiceSelect = document.getElementById('ttsVoiceSelect');
  if (!voiceSelect) return;

  const previousValue = voiceSelect.value;
  voiceSelect.innerHTML = "";

  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "صدای پیش‌فرض سیستم";
  voiceSelect.appendChild(defaultOption);

  // صداهای فارسی را در ابتدای فهرست قرار بده (مشابه tts.html)
  const sortedVoices = [...ttsVoices].sort((a, b) => {
    const aFa = a.lang.toLowerCase().startsWith("fa");
    const bFa = b.lang.toLowerCase().startsWith("fa");
    return Number(bFa) - Number(aFa);
  });

  sortedVoices.forEach((voice) => {
    const option = document.createElement("option");
    option.value = ttsVoices.indexOf(voice);
    option.textContent = `${voice.name} (${voice.lang})${voice.default ? " — پیش‌فرض" : ""}`;
    voiceSelect.appendChild(option);
  });

  if (previousValue !== "" && [...voiceSelect.options].some(o => o.value === previousValue)) {
    voiceSelect.value = previousValue;
  } else {
    // انتخاب خودکار اولین صدای فارسی در صورت وجود
    const firstFaIndex = ttsVoices.findIndex(v => v.lang.toLowerCase().startsWith("fa"));
    if (firstFaIndex !== -1) {
      voiceSelect.value = String(firstFaIndex);
    }
  }

  const persianVoices = ttsVoices.filter(v => v.lang.toLowerCase().startsWith("fa"));
  console.log(`[TTS] ${ttsVoices.length} voices loaded, ${persianVoices.length} Persian voices found.`);
}

function stripVowels(text) {
  if (!text) return "";
  // حذف اعراب و حرکات زبان فارسی و عربی (َ ِ ُ ً ٍ ٌ ّ ْ ٰ) جهت نمایش تمیز در ظاهر فرانت‌اند
  return text.replace(/[ً-ْٰ]/g, "");
}

function cleanTextForSpeech(text) {
  return text
    .replace(/[*_#`~>]/g, "")           // حذف علائم مارک‌داون
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // لینک‌های مارک‌داون به عنوان
    .replace(/^[-•*]\s+/gm, "")         // حذف بولت‌پوینت‌ها
    .replace(/\s+/g, " ")               // نرمال‌سازی فاصله‌ها
    .trim();
}

function splitTextIntoSentences(text) {
  const cleaned = cleanTextForSpeech(text);
  // تقسیم به جملات جهت جلوگیری از باگ کروم در متون طولانی (> 15 ثانیه)
  const parts = cleaned.match(/[^.!?؛\n]+[.!?؛\n]*/g) || [cleaned];
  return parts.map(p => p.trim()).filter(p => p.length > 0);
}

function stopSpeech() {
  speechQueue = [];
  isSpeakingQueue = false;
  if (synth) {
    synth.cancel();
  }
  updateTtsStatus("پخش صدا متوقف شد.");
}

function speakText(text) {
  if (!synth || !("SpeechSynthesisUtterance" in window)) {
    updateTtsStatus("مرورگر شما از Speech Synthesis پشتیبانی نمی‌کند.");
    return;
  }
  if (!text || !text.trim()) return;

  stopSpeech();

  const sentences = splitTextIntoSentences(text);
  if (sentences.length === 0) return;

  speechQueue = [...sentences];
  isSpeakingQueue = true;
  playNextQueueItem();
}

function playNextQueueItem() {
  if (!isSpeakingQueue || speechQueue.length === 0) {
    isSpeakingQueue = false;
    updateTtsStatus("پخش متن تمام شد.");
    return;
  }

  const chunk = speechQueue.shift();
  const utterance = new SpeechSynthesisUtterance(chunk);

  // ذخیره در متغیر سراسری جهت جلوگیری از Garbage Collection در مرورگر کروم
  window._activeUtterance = utterance;

  const voiceSelect = document.getElementById('ttsVoiceSelect');
  const selectedIndex = voiceSelect ? voiceSelect.value : "";

  if (selectedIndex !== "" && ttsVoices[Number(selectedIndex)]) {
    utterance.voice = ttsVoices[Number(selectedIndex)];
    utterance.lang = utterance.voice.lang || "fa-IR";
  } else {
    // تلاش برای یافتن صدای فارسی، در غیر این صورت استفاده از پیش‌فرض
    const faVoice = ttsVoices.find(v => v.lang && v.lang.toLowerCase().startsWith("fa"));
    if (faVoice) {
      utterance.voice = faVoice;
      utterance.lang = faVoice.lang;
    } else {
      utterance.lang = "fa-IR";
    }
  }

  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;

  utterance.onstart = () => {
    const voiceLabel = utterance.voice ? utterance.voice.name : "پیش‌فرض";
    updateTtsStatus(`در حال پخش صدا (${voiceLabel})...`);
  };

  utterance.onend = () => {
    window._activeUtterance = null;
    if (isSpeakingQueue) {
      playNextQueueItem();
    }
  };

  utterance.onerror = (e) => {
    window._activeUtterance = null;
    console.error("TTS Speech Error:", e);
    if (e.error === "canceled" || e.error === "interrupted") {
      isSpeakingQueue = false;
    } else {
      updateTtsStatus(`خطا در پخش صدا (${e.error}) - لطفاً صدای دیگری از منوی کشویی انتخاب کنید.`);
      isSpeakingQueue = false;
    }
  };

  // رفع باگ توقف ناگهانی کروم: فراخوانی resume پیش از speak
  setTimeout(() => {
    try {
      if (synth.paused) {
        synth.resume();
      }
      synth.speak(utterance);
      if (synth.paused) {
        synth.resume();
      }
    } catch (err) {
      console.error("Synth speak error:", err);
    }
  }, 30);
}

/**
 * Send user prompt + extracted camera persona to Shopping Assistant
 */
async function sendPromptToAssistant() {
  const promptInput = document.getElementById('promptInput');
  const sendBtn = document.getElementById('sendBtn');
  const assistantSection = document.getElementById('assistantSection');
  const assistantReply = document.getElementById('assistantReply');
  const intentBadge = document.getElementById('intentBadge');
  const crossSellBox = document.getElementById('crossSellBox');
  const productsGrid = document.getElementById('productsGrid');

  const text = promptInput.value.trim();
  if (!text) {
    alert("لطفاً ابتدا متنی را با صدا بگویید یا تایپ کنید.");
    return;
  }

  stopSpeech();

  sendBtn.disabled = true;
  sendBtn.innerText = "در حال ارتباط با دستیار...";
  assistantSection.classList.remove('hidden');
  assistantReply.innerText = "دستیار در حال بررسی درخواست و جستجو در کاتالوگ محصولات ترب...";
  intentBadge.classList.add('hidden');
  crossSellBox.classList.add('hidden');
  productsGrid.innerHTML = "";

  const profileSourceBadge = document.getElementById('profileSourceBadge');
  if (profileSourceBadge) {
    profileSourceBadge.classList.add('hidden');
    profileSourceBadge.className = 'cache-badge hidden';
  }

  // 1. Identify current user key (identified user ID, or fallback visitor key)
  const userKey = currentIdentity ? currentIdentity.id : "current_visitor";

  let customerProfile = {
    customer_id: currentIdentity ? currentIdentity.id : null,
    customer_name: currentIdentity ? currentIdentity.name : "ناشناس",
    gender: currentGender === "male" ? "مرد" : (currentGender === "female" ? "زن" : null),
    age_group: currentAge ? `${currentAge} سال` : null,
    mood_or_expression: currentEmotion,
    style_preference: "اسپرت / کژوال"
  };

  // 2. Check if user face profile is unchanged and cached within 1 day (Token Cost Optimization)
  const cachedVision = getCachedVisionProfile(userKey);

  if (cachedVision && cachedVision.profile) {
    // Face unchanged & session valid (< 1 day): reuse cached profile without sending picture!
    const cp = cachedVision.profile;
    customerProfile = {
      ...customerProfile,
      ...cp,
      customer_id: currentIdentity ? currentIdentity.id : (cp.customer_id || null),
      customer_name: currentIdentity ? currentIdentity.name : (cp.customer_name || "ناشناس"),
      mood_or_expression: currentEmotion || cp.mood_or_expression
    };

    if (profileSourceBadge) {
      const savedTime = new Date(cachedVision.timestamp).toLocaleTimeString('fa-IR', { hour: '2-digit', minute: '2-digit' });
      profileSourceBadge.className = "cache-badge cached";
      profileSourceBadge.innerText = `⚡ پروفایل بصری کاربر از حافظه کلاینت فراخوانی شد (صرفه‌جویی در هزینه توکن تصویر | ذخیره: ${savedTime})`;
      profileSourceBadge.classList.remove('hidden');
    }
  } else {
    // Session older than 1 day or new profile: capture frame and call Vision Model
    const frameDataUri = captureFrameBase64();
    if (frameDataUri) {
      try {
        const visionRes = await fetch(`${BACKEND_URL}/search/vision/profile`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: frameDataUri,
            customer_id: currentIdentity ? currentIdentity.id : undefined
          })
        });
        if (visionRes.ok) {
          const vData = await visionRes.json();
          customerProfile = {
            ...customerProfile,
            gender: vData.gender || customerProfile.gender,
            age_group: vData.age_group || customerProfile.age_group,
            style_preference: vData.style_preference || customerProfile.style_preference,
            mood_or_expression: currentEmotion || vData.mood_or_expression || customerProfile.mood_or_expression,
            clothing_colors_detected: vData.clothing_colors_detected || [],
            budget_tier: vData.budget_tier || "mid-range",
            extra_attributes: vData.extra_attributes || {},
            summary: vData.summary
          };

          // Store full profile in client-side with timestamp (expires in 1 day)
          setCachedVisionProfile(userKey, customerProfile);

          if (profileSourceBadge) {
            profileSourceBadge.className = "cache-badge fresh";
            profileSourceBadge.innerText = "📸 تحلیل اولیه تصویر با Vision Model انجام و در کلاینت ذخیره شد (اعتبار ۲۴ ساعت)";
            profileSourceBadge.classList.remove('hidden');
          }
        }
      } catch (e) {
        console.warn("Vision model extraction skipped, using local face-api profile:", e);
      }
    }
  }

  // 3. Send user prompt to Agent Chat
  try {
    const chatRes = await fetch(`${BACKEND_URL}/search/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: activeSessionId,
        customer_profile: customerProfile
      })
    });

    if (!chatRes.ok) {
      throw new Error(`خطای سرور: ${chatRes.status}`);
    }

    const data = await chatRes.json();
    activeSessionId = data.session_id;

    // Display reply in UI (vowels stripped for a clean visual appearance)
    lastAssistantReplyText = data.reply; // keeps original vowels for TTS audio playback
    assistantReply.innerText = stripVowels(data.reply);

    // Read result with voice (TTS) using original vocalized text
    if (autoTtsEnabled && data.reply) {
      speakText(data.reply);
    }

    // Display intent
    if (data.search_intent) {
      intentBadge.classList.remove('hidden');
      intentBadge.innerText = `🔍 کوئری فرموله شده برای دیتابیس: «${data.search_intent.search_query || '-'}» | دسته‌بندی: ${data.search_intent.category_filter || 'همه'}`;
    }

    // Display Cross-Sell Offer
    if (data.cross_sell_offer) {
      crossSellBox.classList.remove('hidden');
      crossSellBox.innerHTML = `
        🎁 <strong>پیشنهاد کالای مکمل همراه با تخفیف:</strong><br>
        کالای پیشنهادی: <b>${stripVowels(data.cross_sell_offer.suggested_product_title)}</b><br>
        دلیل تطابق: ${stripVowels(data.cross_sell_offer.reason)}<br>
        تخفیف پکیج: <b>${data.cross_sell_offer.bundle_discount_percent}٪ تخفیف</b> روی خرید همزمان!
      `;
    }

    // Display Products Grid
    if (data.retrieved_products && data.retrieved_products.length > 0) {
      productsGrid.innerHTML = data.retrieved_products.map(p => `
        <div class="product-card">
          <h4>${p.title || 'محصول ترب'}</h4>
          <div class="price">شناسه: ${p.product_id}</div>
          <div style="font-size: 0.8rem; color: #757575; margin-top: 4px;">امتیاز تطابق: ${p.score || '-'}</div>
        </div>
      `).join('');
    }
  } catch (error) {
    console.error("Chat error:", error);
    assistantReply.innerText = `خطا در ارتباط با سرور: ${error.message}. لطفاً مطمئن شوید سرور backend روی پورت ۸۰۰۰ فعال است.`;
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerText = "ارسال به دستیار ترب 🚀";
  }
}

window.onload = async () => {
  await loadModels();
  initSpeechToText();
  initTextToSpeech();

  // Identity UI Controls
  const autoCheckbox = document.getElementById('autoAssignCheckbox');
  if (autoCheckbox) {
    autoAssignNewFaces = autoCheckbox.checked;
    autoCheckbox.addEventListener('change', (e) => {
      autoAssignNewFaces = e.target.checked;
      unknownFaceFrameCount = 0;
    });
  }

  const registerBtn = document.getElementById('registerFaceBtn');
  const nameInput = document.getElementById('manualNameInput');
  if (registerBtn) {
    registerBtn.addEventListener('click', () => {
      const customName = nameInput ? nameInput.value : "";
      const registeredName = registerFace(customName);
      if (registeredName) {
        if (nameInput) nameInput.value = "";
      }
    });
  }

  const toggleManageBtn = document.getElementById('toggleManageBtn');
  const manageDrawer = document.getElementById('manageIdentitiesDrawer');
  if (toggleManageBtn && manageDrawer) {
    toggleManageBtn.addEventListener('click', () => {
      manageDrawer.classList.toggle('hidden');
    });
  }

  const clearBtn = document.getElementById('clearAllIdentitiesBtn');
  if (clearBtn) {
    clearBtn.addEventListener('click', clearAllIdentities);
  }

  document.getElementById('sendBtn').addEventListener('click', sendPromptToAssistant);

  // Press Enter in textarea to send (Ctrl+Enter for newline)
  document.getElementById('promptInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPromptToAssistant();
    }
  });
};
