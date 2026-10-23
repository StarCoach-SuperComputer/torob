// Torob-Yar Assistant - Unified Script matching 1.html, 2.html, 3.html, chat2.html & popup.html

// Global Speech & Emotion State
let recognition = null;
let isRecording = false;
let currentEmotion = "neutral";
let currentAge = 28;
let currentGender = "male";
let activeSessionId = null;
let isCameraLive = false;
let cameraStreamObj = null;

// Identity Detection State (Client-Side Storage)
const STORAGE_KEY = "torob_face_identities";
const VISION_CACHE_KEY = "torob_vision_profiles_cache";
const ONE_DAY_MS = 24 * 60 * 60 * 1000; // 1 day

let storedIdentities = [];
let faceMatcher = null;
let currentIdentity = null; // { id, name, similarity }
let latestDescriptor = null;
let unknownFaceFrameCount = 0;
let autoAssignNewFaces = true;

// Backend API URL
const BACKEND_URL = "http://localhost:8000";

// Mood Map
const moodFaMap = {
  happy: { fa: "خوشحال / باانرژی", icon: "smile", color: "emerald" },
  neutral: { fa: "عادی / آرام", icon: "smile", color: "emerald" },
  sad: { fa: "ناراحت", icon: "frown", color: "amber" },
  surprised: { fa: "متعجب", icon: "sparkles", color: "cyan" },
  angry: { fa: "عصبانی", icon: "alert-circle", color: "rose" },
  fearful: { fa: "نگران", icon: "help-circle", color: "amber" },
  disgusted: { fa: "ناراضی", icon: "alert-triangle", color: "orange" }
};

/**
 * Toast Notification Utility
 */
let toastTimeout;
function showToast(text, isError = false) {
  const toast = document.getElementById('toast');
  const toastText = document.getElementById('toastText');
  const toastIcon = document.getElementById('toastIcon');
  if (!toast || !toastText || !toastIcon) return;

  toastText.textContent = text;
  toastIcon.setAttribute('data-lucide', isError ? 'alert-circle' : 'check-circle');
  toastIcon.className = `w-4 h-4 ${isError ? 'text-red-400' : 'text-emerald-400'}`;
  toast.classList.remove('hidden');

  if (window.lucide) lucide.createIcons();

  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.add('hidden');
  }, 2600);
}

/**
 * Storage & Identity Management (Client-Side)
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
    showToast("هیچ چهره‌ای در تصویر برای ثبت پیدا نشد!", true);
    return null;
  }

  const name = customName && customName.trim() ? customName.trim() : getNextUserLabel();
  const existing = storedIdentities.find(u => u.name === name);

  if (existing) {
    existing.descriptors.push(Array.from(latestDescriptor));
    if (existing.descriptors.length > 5) existing.descriptors.shift();
  } else {
    storedIdentities.push({
      id: `user_${Date.now()}`,
      name: name,
      createdAt: new Date().toISOString(),
      descriptors: [Array.from(latestDescriptor)]
    });
  }

  saveStoredIdentities();
  showToast(`چهره «${name}» با موفقیت ذخیره شد`);
  return name;
}

function deleteIdentity(id) {
  storedIdentities = storedIdentities.filter(u => u.id !== id);
  saveStoredIdentities();
  deleteCachedVisionProfile(id);
  showToast("چهره حذف گردید");
}

function clearAllIdentities() {
  if (confirm("آیا از حذف تمام چهره‌های ذخیره شده اطمینان دارید؟")) {
    storedIdentities = [];
    saveStoredIdentities();
    clearAllVisionCache();
    showToast("تمام چهره‌ها پاک‌سازی شدند");
  }
}

function renderIdentitiesList() {
  const listContainer = document.getElementById('identitiesList');
  const countSpan = document.getElementById('storedCount');
  if (countSpan) countSpan.innerText = storedIdentities.length;
  if (!listContainer) return;

  if (storedIdentities.length === 0) {
    listContainer.innerHTML = '<span class="text-xs text-gray-500">هیچ چهره‌ای هنوز ذخیره نشده است.</span>';
    return;
  }

  listContainer.innerHTML = storedIdentities.map(user => `
    <div class="inline-flex items-center gap-1.5 bg-[#212c3b] border border-[#2d3d52] px-2.5 py-1 rounded-xl text-xs text-white">
      <span>👤 ${user.name}</span>
      <span class="text-gray-400 text-[10px]">(${user.descriptors.length})</span>
      <button onclick="window.onDeleteIdentity('${user.id}')" class="text-red-400 hover:text-red-300 font-bold ml-1 text-sm">&times;</button>
    </div>
  `).join('');
}

window.onDeleteIdentity = (id) => {
  deleteIdentity(id);
};

function updateIdentityUI(identity, noFace = false) {
  const userLabel = document.getElementById('detectedUserLabel');
  const demoLabel = document.getElementById('chatDemographicsLabel');
  const ageGenderText = document.getElementById('currentAgeGenderText');
  const moodText = document.getElementById('currentMoodText');

  const genderFa = currentGender === "male" ? "آقا" : "خانم";
  const mInfo = moodFaMap[currentEmotion] || { fa: currentEmotion, icon: "smile" };

  if (ageGenderText) {
    ageGenderText.innerText = `${genderFa} • ${currentAge || 28} سال`;
  }
  if (moodText) {
    moodText.innerText = mInfo.fa;
  }

  if (noFace) {
    if (userLabel) userLabel.innerText = "در انتظار چهره...";
    if (demoLabel) demoLabel.innerText = "هیچ چهره‌ای در کادر نیست";
  } else if (identity) {
    if (userLabel) userLabel.innerText = `👤 ${identity.name} (${identity.similarity}٪)`;
    if (demoLabel) demoLabel.innerText = `${genderFa} • ${currentAge} سال • ${mInfo.fa}`;
  } else {
    if (userLabel) {
      if (autoAssignNewFaces && unknownFaceFrameCount > 0) {
        userLabel.innerText = `❓ ثبت هویت... (${unknownFaceFrameCount}/5)`;
      } else {
        userLabel.innerText = "❓ کاربر جدید / ناشناس";
      }
    }
    if (demoLabel) demoLabel.innerText = `${genderFa} • ${currentAge} سال • ${mInfo.fa}`;
  }
}

/**
 * Client-Side Vision Profile Cache (1-day TTL)
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
      delete cache[userKey];
      localStorage.setItem(VISION_CACHE_KEY, JSON.stringify(cache));
      return null;
    }
    return entry;
  } catch (e) {
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
  } catch (e) {}
}

function deleteCachedVisionProfile(userKey) {
  try {
    const raw = localStorage.getItem(VISION_CACHE_KEY);
    if (!raw) return;
    const cache = JSON.parse(raw);
    delete cache[userKey];
    localStorage.setItem(VISION_CACHE_KEY, JSON.stringify(cache));
  } catch (e) {}
}

function clearAllVisionCache() {
  try {
    localStorage.removeItem(VISION_CACHE_KEY);
  } catch (e) {}
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
    loadStoredIdentities();
    startRealCamera();
  } catch (error) {
    console.error("Error loading models:", error);
    showToast("خطا در بارگذاری مدل‌های چهره", true);
  }
}

function startRealCamera() {
  const video = document.getElementById('video');
  const pipImg = document.getElementById('pipPortraitImg');

  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 640 } } })
    .then((stream) => {
      cameraStreamObj = stream;
      video.srcObject = stream;
      video.classList.remove('hidden');
      if (pipImg) pipImg.classList.add('hidden');
      isCameraLive = true;
      video.play();
      detectFaces();
      showToast("دوربین زنده فعال شد");
    })
    .catch((error) => {
      console.warn("Camera not granted or unavailable, fallback active:", error);
      isCameraLive = false;
      showToast("حالت شبیه‌ساز پرتره فعال شد");
    });
}

async function detectFaces() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  if (!video || !canvas) return;

  video.addEventListener('loadedmetadata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  });

  const getDisplaySize = () => ({
    width: video.videoWidth || 320,
    height: video.videoHeight || 240,
  });

  canvas.classList.remove('hidden');

  setInterval(async () => {
    if (!isCameraLive || !video.videoWidth || !video.videoHeight) return;

    try {
      const detections = await faceapi.detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, getDisplaySize());
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw subtle mesh/landmarks in red
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

      if (detections[0]) {
        latestDescriptor = detections[0].descriptor;

        const expressions = detections[0].expressions;
        const sorted = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
        currentEmotion = sorted[0][0];
        currentAge = Math.round(detections[0].age);
        currentGender = detections[0].gender;

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
          if (autoAssignNewFaces) {
            unknownFaceFrameCount++;
            updateIdentityUI(null);
            if (unknownFaceFrameCount >= 5) {
              const newName = registerFace();
              unknownFaceFrameCount = 0;
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
        updateIdentityUI(null, true);
      }
    } catch (err) {
      // ignore frame dropped
    }
  }, 400);
}

function captureFrameBase64() {
  const video = document.getElementById('video');
  if (!video || !video.videoWidth || !isCameraLive) return null;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const ctx = tempCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  return tempCanvas.toDataURL('image/jpeg', 0.85);
}

/**
 * Text Cleaners & Speech Synthesis (TTS)
 */
const synth = window.speechSynthesis;
let ttsVoices = [];
let autoTtsEnabled = true;
let lastAssistantReplyText = "";
let speechQueue = [];
let isSpeakingQueue = false;

function stripVowels(text) {
  if (!text) return "";
  return text.replace(/[ً-ْٰ]/g, "");
}

function cleanTextForSpeech(text) {
  return text
    .replace(/[*_#`~>]/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^[-•*]\s+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

function splitTextIntoSentences(text) {
  const cleaned = cleanTextForSpeech(text);
  const parts = cleaned.match(/[^.!?؛\n]+[.!?؛\n]*/g) || [cleaned];
  return parts.map(p => p.trim()).filter(p => p.length > 0);
}

function stopSpeech() {
  speechQueue = [];
  isSpeakingQueue = false;
  if (synth) synth.cancel();
}

function speakText(text) {
  if (!synth || !("SpeechSynthesisUtterance" in window) || !autoTtsEnabled) return;
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
    return;
  }

  const chunk = speechQueue.shift();
  const utterance = new SpeechSynthesisUtterance(chunk);
  window._activeUtterance = utterance;

  if (ttsVoices.length === 0 && synth) {
    ttsVoices = synth.getVoices();
  }
  const faVoice = ttsVoices.find(v => v.lang && v.lang.toLowerCase().startsWith("fa"));
  if (faVoice) {
    utterance.voice = faVoice;
    utterance.lang = faVoice.lang;
  } else {
    utterance.lang = "fa-IR";
  }

  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;

  utterance.onend = () => {
    window._activeUtterance = null;
    if (isSpeakingQueue) playNextQueueItem();
  };

  utterance.onerror = (e) => {
    window._activeUtterance = null;
    if (e.error !== "canceled" && e.error !== "interrupted") {
      isSpeakingQueue = false;
    }
  };

  setTimeout(() => {
    try {
      if (synth.paused) synth.resume();
      synth.speak(utterance);
      if (synth.paused) synth.resume();
    } catch (err) {}
  }, 30);
}

/**
 * Speech Recognition (STT - Speech to Text)
 */
function initSpeechToText() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const micButton = document.getElementById('micButton');
  const micStatusText = document.getElementById('micStatusText');
  const promptInput = document.getElementById('promptInput');
  const langSelect = document.getElementById('langSelect');

  if (!SpeechRecognition) {
    if (micStatusText) micStatusText.innerText = "مرورگر شما از ورودی صوتی پشتیبانی نمی‌کند";
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = langSelect ? langSelect.value : "fa-IR";

  recognition.onstart = () => {
    isRecording = true;
    micButton.classList.add('animate-pulse');
    if (micStatusText) micStatusText.innerText = "در حال شنیدن صدای شما... (صحبت کنید)";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    if (promptInput) promptInput.value = transcript;
  };

  recognition.onerror = (event) => {
    isRecording = false;
    micButton.classList.remove('animate-pulse');
    if (micStatusText) micStatusText.innerText = `خطا در دریافت صدا (${event.error}) - دوباره امتحان کنید`;
  };

  recognition.onend = () => {
    isRecording = false;
    micButton.classList.remove('animate-pulse');
    if (micStatusText) micStatusText.innerText = "برای صحبت لمس کنید (یا کلید Enter)";

    // Auto-send if transcribed
    if (promptInput && promptInput.value.trim().length > 1) {
      sendPromptToAssistant();
    }
  };

  micButton.addEventListener('click', () => {
    stopSpeech();
    if (isRecording) {
      recognition.stop();
    } else {
      if (promptInput) promptInput.value = "";
      recognition.start();
    }
  });
}

/**
 * Chat Stream Rendering Helper
 */
function appendUserMessage(text) {
  const chatFeed = document.getElementById('chatFeed');
  if (!chatFeed) return;

  const bubble = document.createElement('div');
  bubble.className = "flex justify-start";
  bubble.innerHTML = `
    <div class="bg-[#e53945] text-white text-sm font-bold px-5 py-2.5 rounded-2xl shadow-md shadow-red-950/25 max-w-[85%] text-right">
      ${text}
    </div>
  `;
  chatFeed.appendChild(bubble);
  chatFeed.scrollTo({ top: chatFeed.scrollHeight, behavior: 'smooth' });
}

/**
 * Assistant Typing / Loading Indicator in Chat Feed
 */
function showAssistantTypingIndicator(statusText = "دستیار در حال بررسی...") {
  removeAssistantTypingIndicator();
  const chatFeed = document.getElementById('chatFeed');
  if (!chatFeed) return;

  const indicator = document.createElement('div');
  indicator.id = 'assistantTypingIndicator';
  indicator.className = "flex justify-end transition-all duration-200";
  indicator.innerHTML = `
    <div class="bg-[#1b2535] text-gray-200 text-xs px-4 py-3 rounded-2xl max-w-[88%] text-right border border-[#243346] shadow-sm flex items-center gap-3">
      <div class="flex items-center gap-1.5 h-3 flex-shrink-0">
        <span class="w-2 h-2 rounded-full bg-[#e53945] typing-dot"></span>
        <span class="w-2 h-2 rounded-full bg-[#e53945] typing-dot"></span>
        <span class="w-2 h-2 rounded-full bg-[#e53945] typing-dot"></span>
      </div>
      <span class="text-xs text-gray-300 font-medium" id="typingStatusText">${statusText}</span>
    </div>
  `;
  chatFeed.appendChild(indicator);
  chatFeed.scrollTo({ top: chatFeed.scrollHeight, behavior: 'smooth' });
}

function updateAssistantTypingStatus(text) {
  const el = document.getElementById('typingStatusText');
  if (el) el.innerText = text;
}

function removeAssistantTypingIndicator() {
  const indicator = document.getElementById('assistantTypingIndicator');
  if (indicator) {
    indicator.remove();
  }
}

function appendAssistantMessage(rawText, products = [], crossSell = null, cacheBadgeText = null) {
  removeAssistantTypingIndicator();
  const chatFeed = document.getElementById('chatFeed');
  if (!chatFeed) return;

  const cleanDisplay = stripVowels(rawText);

  const container = document.createElement('div');
  container.className = "space-y-3";

  // Text Reply Card
  let cacheBadgeHtml = cacheBadgeText ? `
    <div class="mt-2 text-[10px] text-emerald-400 bg-emerald-950/50 border border-emerald-800/40 px-2 py-1 rounded-lg inline-block">
      ${cacheBadgeText}
    </div>
  ` : '';

  container.innerHTML = `
    <div class="flex justify-end">
      <div class="bg-[#1b2535] text-gray-100 text-[13.5px] leading-relaxed font-normal p-4 rounded-2xl max-w-[88%] text-right border border-[#243346] shadow-sm">
        ${cleanDisplay}
        ${cacheBadgeHtml}
      </div>
    </div>
  `;

  // Cross-sell offer banner if present
  if (crossSell) {
    const csBox = document.createElement('div');
    csBox.className = "flex justify-end";
    csBox.innerHTML = `
      <div class="bg-[#241c1a] border border-[#522b28] rounded-2xl p-3.5 max-w-[88%] text-right text-xs text-orange-200">
        <div class="font-bold text-red-400 mb-1">🎁 پیشنهاد کالای مکمل با تخفیف جفتی:</div>
        <div>${stripVowels(crossSell.reason)}</div>
        <div class="mt-1 font-bold text-white">کالای مکمل: ${stripVowels(crossSell.suggested_product_title)} (${crossSell.bundle_discount_percent}٪ تخفیف خرید همزمان)</div>
      </div>
    `;
    container.appendChild(csBox);
  }

  // Products Cards (Design from chat2.html with real shops and offers from data/ via repository)
  if (products && products.length > 0) {
    products.forEach((p, idx) => {
      const pTitle = p.title || 'محصول ترب';
      const pScore = p.score ? `امتیاز: ${(p.score * 100).toFixed(0)}٪` : 'تطابق عالی';
      const pPrice = p.min_price_formatted || (p.min_price ? `از ${Number(p.min_price).toLocaleString('fa-IR')} تومان` : 'مشاهده قیمت در ترب');
      const pImage = p.image_url || p.primary_image || (idx % 2 === 0
        ? "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=350&q=80"
        : "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?auto=format&fit=crop&w=350&q=80");

      const offersCount = p.offers_count || (p.shops ? p.shops.length : 0);
      const sellersCountText = offersCount > 0 ? `${offersCount} فروشگاه در ترب` : 'موجود در فروشگاه‌های ترب';

      // Real top shop badge from repository
      const topShop = (p.shops && p.shops.length > 0) ? p.shops[0] : null;
      const topShopBadge = topShop ? `
        <span class="text-[10px] text-emerald-400 font-semibold bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded-md flex items-center gap-1">
          <i data-lucide="store" class="w-3 h-3"></i>
          <span>${topShop.shop_name}</span>
        </span>
      ` : '';

      const card = document.createElement('div');
      card.className = "w-full bg-[#182230] hover:bg-[#1c2838] border border-[#263447] rounded-[22px] p-3 flex items-center justify-between transition-colors shadow-md press-action cursor-pointer";
      card.onclick = () => openProductDetailModal(p);

      card.innerHTML = `
        <div class="flex flex-col items-start justify-center gap-1.5 pr-2 text-right">
          <h3 class="text-[13px] font-bold text-gray-100 leading-snug">${pTitle}</h3>
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-black text-[#eb445a]">${pPrice}</span>
            <span class="text-[10px] text-gray-400 font-medium">(${pScore})</span>
          </div>
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="text-[11px] text-gray-400 font-medium">${sellersCountText}</span>
            ${topShopBadge}
          </div>
          <div class="pt-0.5 flex items-center gap-1 text-[#e53945] text-xs font-bold">
            <i data-lucide="chevron-left" class="w-3.5 h-3.5"></i>
            <span>مشاهده فروشگاه‌ها و خرید</span>
          </div>
        </div>
        <div class="w-20 h-20 sm:w-24 sm:h-24 rounded-2xl bg-white p-1.5 flex items-center justify-center flex-shrink-0 shadow-inner overflow-hidden">
          <img src="${pImage}" alt="${pTitle}" class="w-full h-full object-contain filter contrast-105" loading="lazy" />
        </div>
      `;
      container.appendChild(card);
    });
  }

  chatFeed.appendChild(container);
  if (window.lucide) lucide.createIcons();
  chatFeed.scrollTo({ top: chatFeed.scrollHeight, behavior: 'smooth' });
}

/**
 * Send user prompt + extracted camera persona to Shopping Assistant
 */
async function sendPromptToAssistant() {
  const promptInput = document.getElementById('promptInput');
  const sendBtn = document.getElementById('sendBtn');
  const text = promptInput ? promptInput.value.trim() : "";

  if (!text) {
    showToast("لطفاً ابتدا متنی را بگویید یا تایپ کنید", true);
    return;
  }

  stopSpeech();
  appendUserMessage(text);
  showAssistantTypingIndicator("دستیار در حال بررسی...");
  if (promptInput) promptInput.value = "";

  if (sendBtn) sendBtn.disabled = true;
  const headerStatus = document.getElementById('headerStatusText');
  if (headerStatus) headerStatus.innerText = "دستیار در حال بررسی...";

  // 1. Identify user and profile
  const userKey = currentIdentity ? currentIdentity.id : "current_visitor";
  let customerProfile = {
    customer_id: currentIdentity ? currentIdentity.id : null,
    customer_name: currentIdentity ? currentIdentity.name : "ناشناس",
    gender: currentGender === "male" ? "مرد" : (currentGender === "female" ? "زن" : null),
    age_group: currentAge ? `${currentAge} سال` : null,
    mood_or_expression: currentEmotion,
    style_preference: "اسپرت / کژوال"
  };

  // 2. Check 1-day Client Cache for Vision Model
  let cacheStatusBadge = null;
  const cachedVision = getCachedVisionProfile(userKey);

  if (cachedVision && cachedVision.profile) {
    const cp = cachedVision.profile;
    customerProfile = {
      ...customerProfile,
      ...cp,
      customer_id: currentIdentity ? currentIdentity.id : (cp.customer_id || null),
      customer_name: currentIdentity ? currentIdentity.name : (cp.customer_name || "ناشناس"),
      mood_or_expression: currentEmotion || cp.mood_or_expression
    };
    cacheStatusBadge = "⚡ پروفایل از حافظه کلاینت فراخوانی شد (صرفه‌جویی توکن تصویر)";
  } else {
    const frameDataUri = captureFrameBase64();
    if (frameDataUri) {
      updateAssistantTypingStatus("در حال تحلیل استایل و چهره با هوش مصنوعی...");
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
          setCachedVisionProfile(userKey, customerProfile);
          cacheStatusBadge = "📸 تحلیل تصویر انجام شد (ذخیره تا ۲۴ ساعت)";
        }
      } catch (e) {
        console.warn("Vision model skipped, fallback to face-api:", e);
      }
    }
  }

  // 3. Send to /search/chat
  updateAssistantTypingStatus("در حال جستجوی بهترین پیشنهاد در ترب...");
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

    lastAssistantReplyText = data.reply;
    removeAssistantTypingIndicator();
    appendAssistantMessage(data.reply, data.retrieved_products, data.cross_sell_offer, cacheStatusBadge);

    // Read loud with voice using original vocalized response
    if (autoTtsEnabled && data.reply) {
      speakText(data.reply);
    }
  } catch (error) {
    console.error("Chat error:", error);
    removeAssistantTypingIndicator();
    appendAssistantMessage(`خطا در ارتباط با سرور: ${error.message}. لطفاً مطمئن شوید بک‌اند روی پورت ۸۰۰۰ فعال است.`);
  } finally {
    removeAssistantTypingIndicator();
    if (sendBtn) sendBtn.disabled = false;
    if (headerStatus) headerStatus.innerText = "آنلاین و آماده گفتگو";
  }
}

/**
 * Product Detail Modal (populated with actual shops and offers from data/ via repository)
 */
async function openProductDetailModal(product, optionalPrice, optionalImg) {
  const modal = document.getElementById('productDetailModal');
  const dTitle = document.getElementById('detailModalTitle');
  const dPrice = document.getElementById('detailModalPrice');
  const dImg = document.getElementById('detailModalImg');
  const dSellersCount = document.getElementById('detailModalSellersCount');
  const sellersList = document.getElementById('detailModalSellersList');
  const specsList = document.getElementById('detailModalSpecsList');

  // Support both object parameter and legacy (title, price, img)
  let p = {};
  if (typeof product === "object" && product !== null) {
    p = product;
  } else {
    p = {
      title: product || 'محصول ترب',
      min_price_formatted: optionalPrice || 'استعلام قیمت در ترب',
      image_url: optionalImg || '',
      shops: [],
    };
  }

  const title = p.title || p.persian_title || 'محصول ترب';
  const price = p.min_price_formatted || (p.min_price ? `از ${Number(p.min_price).toLocaleString('fa-IR')} تومان` : (optionalPrice || 'استعلام قیمت'));
  const fallbackImg = "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=700&q=85";
  const imgUrl = p.image_url || p.primary_image || optionalImg || fallbackImg;

  if (dTitle) dTitle.innerText = title;
  if (dPrice) dPrice.innerText = price;
  if (dImg) dImg.src = imgUrl;

  // Render product specs if present
  if (specsList) {
    if (p.specs && p.specs.length > 0) {
      specsList.innerHTML = p.specs.slice(0, 4).map(s => `
        <span class="bg-[#212c3b] text-gray-200 text-xs px-2.5 py-1 rounded-lg border border-[#2b3a4c]">${s.key}: ${s.value}</span>
      `).join('');
    } else {
      specsList.innerHTML = `
        <span class="bg-[#212c3b] text-gray-200 text-xs px-3 py-1 rounded-lg">گارانتی اصالت کالا</span>
        <span class="bg-[#e53945] text-white text-xs font-bold px-3 py-1 rounded-lg">ارسال فوری</span>
        <span class="bg-[#212c3b] text-gray-200 text-xs px-3 py-1 rounded-lg">۷ روز ضمانت بازگشت</span>
      `;
    }
  }

  let shops = p.shops || [];

  // If shops are not yet in memory but we have product_id, fetch from backend endpoint
  if ((!shops || shops.length === 0) && p.product_id) {
    if (sellersList) {
      sellersList.innerHTML = `
        <div class="p-4 text-center text-xs text-gray-400 animate-pulse">
          در حال دریافت اطلاعات فروشگاه‌ها و قیمت‌های لحظه‌ای ترب...
        </div>
      `;
    }
    if (modal) modal.classList.remove('hidden');

    try {
      const res = await fetch(`${BACKEND_URL}/search/products/${p.product_id}/details`);
      if (res.ok) {
        const details = await res.json();
        shops = details.shops || [];
        if (details.min_price_formatted && dPrice) dPrice.innerText = details.min_price_formatted;
        if (details.primary_image && dImg) dImg.src = details.primary_image;
        if (details.specs && specsList && details.specs.length > 0) {
          specsList.innerHTML = details.specs.slice(0, 4).map(s => `
            <span class="bg-[#212c3b] text-gray-200 text-xs px-2.5 py-1 rounded-lg border border-[#2b3a4c]">${s.key}: ${s.value}</span>
          `).join('');
        }
      }
    } catch (err) {
      console.warn("Could not fetch product details from server:", err);
    }
  }

  if (dSellersCount) {
    dSellersCount.innerText = shops.length > 0 ? `${shops.length} فروشگاه فعال در ترب` : 'فروشگاه‌های معتبر ترب';
  }

  // Populate sellers list with real shops from Shops repository
  if (sellersList) {
    if (shops.length === 0) {
      sellersList.innerHTML = `
        <div class="bg-[#1b2634] border border-[#253447] rounded-2xl p-4 text-center text-xs text-gray-400">
          فروشگاه‌های همکار این کالا در حال به‌روزرسانی هستند.
        </div>
      `;
    } else {
      sellersList.innerHTML = shops.map((s, idx) => {
        const sName = s.shop_name || s.name || `فروشگاه ترب #${idx + 1}`;
        const sPrice = s.price_formatted || (s.price > 0 ? `${Number(s.price).toLocaleString('fa-IR')} تومان` : 'استعلام قیمت');
        const sCity = s.city || s.location_label || s.address || 'تهران';
        const sWarranty = s.warranty || 'گارانتی اصالت ترب';
        const isFastDelivery = Boolean(s.supports_fast_delivery);
        const webUrl = s.website_url || '';

        const deliveryBadge = isFastDelivery ? `
          <span class="bg-emerald-950/70 text-emerald-400 text-[10px] font-bold px-2 py-0.5 rounded-full border border-emerald-800/40">
            ارسال فوری
          </span>
        ` : `
          <span class="bg-[#1e2938] text-gray-300 text-[10px] font-medium px-2 py-0.5 rounded-full border border-[#2a3a4f]">
            ارسال سراسری
          </span>
        `;

        return `
          <article class="bg-[#1b2634] hover:bg-[#202c3c] border border-[#253447] rounded-2xl p-3.5 flex items-center justify-between text-right transition-colors shadow-sm">
            <!-- Left Side: Price & Purchase Action -->
            <div class="flex flex-col items-start gap-2">
              <span class="text-xs sm:text-[13px] font-bold text-white tracking-normal font-sans">
                ${sPrice}
              </span>
              <button
                onclick="confirmPurchaseStore('${sName}', '${sPrice}', '${webUrl}')"
                class="bg-[#e53945] hover:bg-[#d42d39] active:scale-95 text-white text-[11px] font-bold py-1.5 px-3.5 rounded-xl shadow-md shadow-red-950/30 transition-all press-action whitespace-nowrap"
              >
                مشاهده و خرید
              </button>
            </div>

            <!-- Right Side: Store Title, Torob Badges, Location -->
            <div class="flex flex-col items-end">
              <h4 class="text-[13px] font-bold text-gray-100">${sName}</h4>
              <div class="flex items-center gap-1.5 mt-1 flex-wrap justify-end">
                ${deliveryBadge}
                <span class="text-[10px] text-gray-400">${sWarranty}</span>
              </div>
              <span class="text-[10.5px] text-gray-400 mt-1 flex items-center gap-1">
                <i data-lucide="map-pin" class="w-3 h-3 text-gray-500"></i>
                <span>${sCity}</span>
              </span>
            </div>
          </article>
        `;
      }).join('');
    }
  }

  if (window.lucide) lucide.createIcons();
  if (modal) modal.classList.remove('hidden');
}

function closeProductDetailModal() {
  const modal = document.getElementById('productDetailModal');
  if (modal) modal.classList.add('hidden');
}

window.confirmPurchaseStore = (storeName, storePrice, websiteUrl) => {
  closeProductDetailModal();
  if (websiteUrl && websiteUrl !== 'null' && websiteUrl !== 'undefined' && websiteUrl.startsWith('http')) {
    showToast(`در حال انتقال به وبسایت رسمی ${storeName}...`);
    setTimeout(() => {
      window.open(websiteUrl, '_blank');
    }, 1200);
  } else {
    showToast(`در حال انتقال به درگاه مستقیم خرید از ${storeName} در ترب...`);
  }
};

/**
 * Window Onload Initialization
 */
window.onload = async () => {
  if (window.lucide) lucide.createIcons();

  await loadModels();
  initSpeechToText();

  // TTS mute toggle button
  const muteBtn = document.getElementById('toggleTtsMuteBtn');
  const muteIcon = document.getElementById('ttsMuteIcon');
  if (muteBtn) {
    muteBtn.addEventListener('click', () => {
      autoTtsEnabled = !autoTtsEnabled;
      if (!autoTtsEnabled) {
        stopSpeech();
        muteIcon.setAttribute('data-lucide', 'volume-x');
        showToast("خواندن صوتی غیرفعال شد");
      } else {
        muteIcon.setAttribute('data-lucide', 'volume-2');
        showToast("خواندن صوتی فعال شد");
      }
      if (window.lucide) lucide.createIcons();
    });
  }

  // Camera toggle button on PiP
  const toggleCamBtn = document.getElementById('toggleCamBtn');
  const pipImg = document.getElementById('pipPortraitImg');
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');

  if (toggleCamBtn) {
    toggleCamBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!isCameraLive) {
        startRealCamera();
      } else {
        if (cameraStreamObj) {
          cameraStreamObj.getTracks().forEach(t => t.stop());
        }
        video.classList.add('hidden');
        canvas.classList.add('hidden');
        pipImg.classList.remove('hidden');
        isCameraLive = false;
        showToast("حالت شبیه‌ساز پرتره فعال شد");
      }
    });
  }

  // Quick Register Face Button on PiP
  const quickRegBtn = document.getElementById('quickRegisterBtn');
  if (quickRegBtn) {
    quickRegBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      registerFace();
    });
  }

  // Privacy Popup & Modal (popup.html)
  const privacyBtn = document.getElementById('privacyPopupBtn');
  const privacyModal = document.getElementById('privacyModal');
  const closePrivacyBtn = document.getElementById('closePrivacyBtn');
  const confirmConsentBtn = document.getElementById('confirmCameraConsentBtn');
  const voiceOnlyBtn = document.getElementById('voiceOnlyBtn');

  if (privacyBtn && privacyModal) {
    privacyBtn.addEventListener('click', () => privacyModal.classList.remove('hidden'));
  }
  if (closePrivacyBtn && privacyModal) {
    closePrivacyBtn.addEventListener('click', () => privacyModal.classList.add('hidden'));
  }
  if (voiceOnlyBtn && privacyModal) {
    voiceOnlyBtn.addEventListener('click', () => {
      privacyModal.classList.add('hidden');
      showToast("حالت فقط صوتی انتخاب شد");
    });
  }
  if (confirmConsentBtn && privacyModal) {
    confirmConsentBtn.addEventListener('click', () => {
      privacyModal.classList.add('hidden');
      showToast("تنظیمات حریم خصوصی تایید شد");
      if (!isCameraLive) startRealCamera();
    });
  }

  // Identities Management Drawer/Modal
  const toggleManageBtn = document.getElementById('toggleManageBtn');
  const identitiesModal = document.getElementById('identitiesModal');
  const closeIdentitiesBtn = document.getElementById('closeIdentitiesBtn');
  const registerFaceBtn = document.getElementById('registerFaceBtn');
  const manualNameInput = document.getElementById('manualNameInput');
  const clearAllBtn = document.getElementById('clearAllIdentitiesBtn');
  const autoCheckbox = document.getElementById('autoAssignCheckbox');

  if (toggleManageBtn && identitiesModal) {
    toggleManageBtn.addEventListener('click', () => identitiesModal.classList.remove('hidden'));
  }
  if (closeIdentitiesBtn && identitiesModal) {
    closeIdentitiesBtn.addEventListener('click', () => identitiesModal.classList.add('hidden'));
  }
  if (registerFaceBtn) {
    registerFaceBtn.addEventListener('click', () => {
      const name = manualNameInput ? manualNameInput.value : "";
      registerFace(name);
      if (manualNameInput) manualNameInput.value = "";
    });
  }
  if (clearAllBtn) {
    clearAllBtn.addEventListener('click', clearAllIdentities);
  }
  if (autoCheckbox) {
    autoCheckbox.addEventListener('change', (e) => {
      autoAssignNewFaces = e.target.checked;
      showToast(autoAssignNewFaces ? "ثبت خودکار فعال شد" : "ثبت خودکار غیرفعال شد");
    });
  }

  // Send message bindings
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) {
    sendBtn.addEventListener('click', sendPromptToAssistant);
  }

  const promptInput = document.getElementById('promptInput');
  if (promptInput) {
    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        sendPromptToAssistant();
      }
    });
  }
};
