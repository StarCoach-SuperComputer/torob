// Global state
let recognition = null;
let isRecording = false;
let currentEmotion = "neutral";
let currentAge = null;
let currentGender = null;
let activeSessionId = null;

// Backend API URL
const BACKEND_URL = "http://localhost:8000";

async function loadModels() {
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromUri('models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('models');
    await faceapi.nets.ageGenderNet.loadFromUri('models');
    await faceapi.nets.faceExpressionNet.loadFromUri('models');

    console.log("Models loaded successfully.");
    document.getElementById('loadingIndicator').style.display = 'none';

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
      detectEmotions();
    })
    .catch((error) => {
      console.error("Error accessing webcam:", error);
      warningMessage.classList.remove('hidden');
    });
}

async function detectEmotions() {
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

    const detections = await faceapi.detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender();

    const resizedDetections = faceapi.resizeResults(detections, getDisplaySize());

    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    if (detections[0]) {
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
    } else {
      emotionResult.innerText = "هیچ چهره‌ای در کادر دوربین شناسایی نشد.";
    }
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

  sendBtn.disabled = true;
  sendBtn.innerText = "در حال ارتباط با دستیار...";
  assistantSection.classList.remove('hidden');
  assistantReply.innerText = "دستیار در حال بررسی درخواست و جستجو در کاتالوگ محصولات ترب...";
  intentBadge.classList.add('hidden');
  crossSellBox.classList.add('hidden');
  productsGrid.innerHTML = "";

  // 1. First, try to extract complete vision persona if frame available
  let customerProfile = {
    gender: currentGender === "male" ? "مرد" : (currentGender === "female" ? "زن" : null),
    age_group: currentAge ? `${currentAge} سال` : null,
    mood_or_expression: currentEmotion,
    style_preference: "اسپرت / کژوال"
  };

  const frameDataUri = captureFrameBase64();
  if (frameDataUri) {
    try {
      const visionRes = await fetch(`${BACKEND_URL}/search/vision/profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frameDataUri })
      });
      if (visionRes.ok) {
        const vData = await visionRes.json();
        customerProfile = {
          gender: vData.gender || customerProfile.gender,
          age_group: vData.age_group || customerProfile.age_group,
          style_preference: vData.style_preference || customerProfile.style_preference,
          mood_or_expression: vData.mood_or_expression || customerProfile.mood_or_expression,
          clothing_colors_detected: vData.clothing_colors_detected || [],
          budget_tier: vData.budget_tier || "mid-range",
          extra_attributes: vData.extra_attributes || {}
        };
      }
    } catch (e) {
      console.warn("Vision model extraction skipped, using local face-api profile:", e);
    }
  }

  // 2. Send user prompt to Agent Chat
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

    // Display reply
    assistantReply.innerText = data.reply;

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
        کالای پیشنهادی: <b>${data.cross_sell_offer.suggested_product_title}</b><br>
        دلیل تطابق: ${data.cross_sell_offer.reason}<br>
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

  document.getElementById('sendBtn').addEventListener('click', sendPromptToAssistant);

  // Press Enter in textarea to send (Ctrl+Enter for newline)
  document.getElementById('promptInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPromptToAssistant();
    }
  });
};
