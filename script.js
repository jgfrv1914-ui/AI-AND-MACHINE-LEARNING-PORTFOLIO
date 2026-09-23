/* ============================================================
   CONFIGURATION - the only block you need to edit when deploying
   ============================================================
   When you deploy a demo (Streamlit Community Cloud or Hugging
   Face Spaces), paste its URL here. While a demo is left empty,
   its card shows "Run locally" and links to the project README
   instead of a dead link.
   ============================================================ */
const CONFIG = {
  github: "https://github.com/jgfrv1914-ui/AI-AND-MACHINE-LEARNING-PORTFOLIO",
  projects: {
    housing: { folder: "01_housing_price_prediction", demo: "" },
    text:    { folder: "02_text_classification",      demo: "" },
    fraud:   { folder: "03_fraud_detection",          demo: "" },
  },
};

const TEXTS = {
  es: {
    nav: ["Proyectos", "Metodología", "Certificaciones", "Sobre mí", "Hablemos"],
    copied: 'Copiado',
    demo: 'Abrir demo <span aria-hidden="true">↗</span>',
    local: 'Ejecutar en local <span aria-hidden="true">↗</span>',
    code: 'Ver código <span aria-hidden="true">↗</span>',
    blocks: {
      "[data-section='hero'] .eyebrow": '<span class="status-dot"></span> Disponible para proyectos de IA',
      "[data-section='hero'] h1": "Convierto datos<br /><em>en decisiones.</em>",
      "[data-section='hero'] .hero-lede": "Soy <strong>Fernando Tafurt Pinto</strong>, especialista en formación en Inteligencia Artificial y Machine Learning. Construyo modelos que resuelven problemas reales y publico sus métricas sin maquillar.",
      "[data-section='hero'] .button-primary": 'Ver mis proyectos <span aria-hidden="true">↓</span>',
      "[data-section='hero'] .button-ghost": 'Conocerme mejor <span aria-hidden="true">↗</span>',
      "[data-section='projects'] .eyebrow": "01 / Proyectos seleccionados",
      "[data-section='projects'] h2": "Problemas reales.<br /><em>Métricas honestas.</em>",
      "[data-section='projects'] .section-note": "Tres familias distintas de problema: regresión tabular, clasificación de texto y clasificación con clases desbalanceadas. Cada uno con su código, sus métricas y sus limitaciones documentadas.",
      "[data-project='housing'] h3": "Predicción de precios de vivienda",
      "[data-project='housing'] .project-body > p": "Regresión sobre 20.640 distritos censales de California. Comparación de tres modelos y demo interactiva.",
      "[data-project='text'] h3": "Clasificador de texto por tema",
      "[data-project='text'] .project-body > p": "Clasificación de mensajes en 20 categorías. Explica qué términos concretos empujaron cada decisión.",
      "[data-project='fraud'] h3": "Detección de fraude",
      "[data-project='fraud'] .project-body > p": "Clases muy desbalanceadas sobre 284.807 transacciones reales. Umbral elegido en validación y explicación con SHAP.",
      "[data-section='method'] .eyebrow": "02 / Cómo trabajo",
      "[data-section='method'] h2": "Una métrica solo vale<br /><em>si es honesta.</em>",
      "[data-section='method'] .section-note": "Es fácil inflar un resultado sin darse cuenta. Estas son las tres reglas que sigo en los tres proyectos, y son la razón de que mis cifras sean más bajas de lo que podrían parecer.",
      "[data-section='certs'] .eyebrow": "03 / Certificaciones",
      "[data-section='certs'] h2": "Formación que<br /><em>se puede verificar.</em>",
      "[data-section='certs'] .section-note": "Once credenciales de cinco emisores. Cada una enlaza a su página oficial de verificación, salvo las dos que sus emisores entregan únicamente como documento y que se muestran completas.",
      "[data-cert='ml'] .cert-lede": "La especialización de Andrew Ng: los fundamentos sobre los que se apoyan los tres proyectos de este portafolio. Unas 94 horas entre los tres cursos.",
      "[data-cert='ml'] .cert-main-link": 'Verificar la especialización <span aria-hidden="true">↗</span>',
      "[data-cert='anthropic'] .cert-lede": "Cuatro cursos sobre uso y construcción con modelos de lenguaje, desde los fundamentos hasta el desarrollo contra la API.",
      "[data-cert='huawei'] .cert-lede": "Curso de certificación en Inteligencia Artificial de Huawei: fundamentos de machine learning y deep learning con el marco MindSpore.",
      "[data-cert='huawei'] .cert-thumb > span": 'Ver el certificado <span aria-hidden="true">↗</span>',
      "[data-cert='talentotech'] h3": "IA &middot; Nivel Intermedio",
      "[data-cert='talentotech'] .cert-lede": "Bootcamp de <b>159 horas</b> del MinTIC, impartido por la Unión Temporal IU Training: Universidad de Antioquia, Universidad de Caldas y Ubicua Technology.",
      "[data-cert='talentotech'] .cert-thumb > span": 'Ver el certificado <span aria-hidden="true">↗</span>',
      "[data-cert='aws'] .cert-lede": "Insignia de AWS Educate: conceptos fundamentales de machine learning y aplicación del pipeline completo a un problema de negocio, con evaluación posterior al curso.",
      "[data-cert='aws'] .cert-main-link": 'Verificar en Credly <span aria-hidden="true">↗</span>',
      "[data-section='about'] .eyebrow": "04 / Sobre mí",
      "[data-section='about'] h2": "La curiosidad es<br /><em>mi combustible.</em>",
      "[data-section='about'] .about-copy > p:not(.eyebrow)": "Me interesa entender cómo los datos pueden ayudarnos a tomar mejores decisiones. Mi trabajo combina fundamentos sólidos de Machine Learning, comunicación clara y el deseo de construir soluciones que generen impacto.",
      "[data-section='contact'] .eyebrow": "05 / Contacto",
      "[data-section='contact'] h2": "¿Construimos algo<br /><em>inteligente?</em>",
      "[data-section='contact'] .contact-inner > p:not(.eyebrow)": "Estoy abierto a colaboraciones, oportunidades y conversaciones sobre datos e Inteligencia Artificial.",
      "[data-section='contact'] .button-primary": 'Escríbeme un correo <span aria-hidden="true">↗</span>',
      ".skills-list": "<span>01 &middot; Python &amp; SQL</span><span>02 &middot; Modelado predictivo</span><span>03 &middot; NLP</span><span>04 &middot; Visualización</span>",
      "[data-project='housing'] .project-label": "REGRESIÓN &middot; XGBOOST",
      "[data-project='fraud'] .project-label": "CLASIFICACIÓN &middot; SHAP",
      "[data-project='housing'] .art-caption": "California Housing &middot; R² en test",
      "[data-project='text'] .art-caption": "20 Newsgroups &middot; F1 macro (20 clases)",
      "[data-project='fraud'] .art-caption": "Credit Card Fraud &middot; 0,17 % positivos",
      "[data-project='fraud'] .fraud-stat small": "precisión en test",
      ".contact-copy": "Copiar",
      ".footer-linkedin": "LinkedIn ↗",
      ".footer a:not(.footer-linkedin)": "Volver arriba ↑",
    },
    methods: [
      ["El test se toca una sola vez", "Partición 60/20/20. El modelo, los hiperparámetros y el umbral se eligen mirando <b>validación</b>. El test se usa al final, con el ganador, y no interviene en ninguna decisión."],
      ["Cada número con su desviación", "Una cifra suelta de una sola partición no dice nada. Publico validación cruzada con su desviación: <b>R² 0.847 ± 0.007</b> es más útil, y más honesto, que un 0.91 afortunado."],
      ["La métrica correcta para el problema", "Con un 0,17 % de fraude, ROC-AUC se ve bien casi siempre y engaña. Por eso el proyecto 03 se juzga con <b>PR-AUC</b>, y el 02 con <b>F1 macro</b> en lugar de exactitud."],
    ],
    metrics: {
      housing: [["R² test", "0.850"], ["R² val. cruzada", "0.847 ± 0.007"], ["MAE", "≈ $29.600"]],
      text: [["F1 macro", "0.706"], ["Exactitud", "0.719"], ["Azar", "0.050"]],
      fraud: [["Precisión", "0.931"], ["Recall", "0.818"], ["PR-AUC", "0.874"]],
    },
  },

  en: {
    nav: ["Projects", "Methodology", "Certifications", "About me", "Let’s talk"],
    copied: 'Copied',
    demo: 'Open demo <span aria-hidden="true">↗</span>',
    local: 'Run locally <span aria-hidden="true">↗</span>',
    code: 'View code <span aria-hidden="true">↗</span>',
    blocks: {
      "[data-section='hero'] .eyebrow": '<span class="status-dot"></span> Available for AI projects',
      "[data-section='hero'] h1": "I turn data<br /><em>into decisions.</em>",
      "[data-section='hero'] .hero-lede": "I’m <strong>Fernando Tafurt Pinto</strong>, an AI and Machine Learning professional in training. I build models that solve real problems and I publish their metrics unpolished.",
      "[data-section='hero'] .button-primary": 'View my projects <span aria-hidden="true">↓</span>',
      "[data-section='hero'] .button-ghost": 'Get to know me <span aria-hidden="true">↗</span>',
      "[data-section='projects'] .eyebrow": "01 / Selected projects",
      "[data-section='projects'] h2": "Real problems.<br /><em>Honest metrics.</em>",
      "[data-section='projects'] .section-note": "Three different families of problem: tabular regression, text classification and classification with imbalanced classes. Each one with its code, its metrics and its limitations documented.",
      "[data-project='housing'] h3": "Housing price prediction",
      "[data-project='housing'] .project-body > p": "Regression over 20,640 California census districts. Three models compared plus an interactive demo.",
      "[data-project='text'] h3": "Topic classifier for text",
      "[data-project='text'] .project-body > p": "Message classification across 20 categories. It shows which specific terms drove each decision.",
      "[data-project='fraud'] h3": "Fraud detection",
      "[data-project='fraud'] .project-body > p": "Heavily imbalanced classes over 284,807 real transactions. Threshold picked on validation and explained with SHAP.",
      "[data-section='method'] .eyebrow": "02 / How I work",
      "[data-section='method'] h2": "A metric only counts<br /><em>if it is honest.</em>",
      "[data-section='method'] .section-note": "It is easy to inflate a result without noticing. These are the three rules I follow across all three projects, and the reason my numbers are lower than they could look.",
      "[data-section='certs'] .eyebrow": "03 / Certifications",
      "[data-section='certs'] h2": "Training you<br /><em>can verify.</em>",
      "[data-section='certs'] .section-note": "Eleven credentials from five issuers. Every one links to its official verification page, except the two that their issuers only hand over as a document, shown here in full.",
      "[data-cert='ml'] .cert-lede": "Andrew Ng’s specialization: the foundations the three projects in this portfolio rest on. Roughly 94 hours across the three courses.",
      "[data-cert='ml'] .cert-main-link": 'Verify the specialization <span aria-hidden="true">↗</span>',
      "[data-cert='anthropic'] .cert-lede": "Four courses on using and building with language models, from the fundamentals through to developing against the API.",
      "[data-cert='huawei'] .cert-lede": "Huawei’s Artificial Intelligence certification course: machine learning and deep learning fundamentals with the MindSpore framework.",
      "[data-cert='huawei'] .cert-thumb > span": 'View the certificate <span aria-hidden="true">↗</span>',
      "[data-cert='talentotech'] h3": "AI &middot; Intermediate Level",
      "[data-cert='talentotech'] .cert-lede": "A <b>159-hour</b> MinTIC bootcamp, delivered by the IU Training consortium: Universidad de Antioquia, Universidad de Caldas and Ubicua Technology.",
      "[data-cert='talentotech'] .cert-thumb > span": 'View the certificate <span aria-hidden="true">↗</span>',
      "[data-cert='aws'] .cert-lede": "AWS Educate badge: machine learning fundamentals and applying the full pipeline to a business problem, with a post-course assessment.",
      "[data-cert='aws'] .cert-main-link": 'Verify on Credly <span aria-hidden="true">↗</span>',
      "[data-section='about'] .eyebrow": "04 / About me",
      "[data-section='about'] h2": "Curiosity is<br /><em>my fuel.</em>",
      "[data-section='about'] .about-copy > p:not(.eyebrow)": "I’m interested in understanding how data can help us make better decisions. My work combines strong Machine Learning foundations, clear communication and a desire to build solutions that create impact.",
      "[data-section='contact'] .eyebrow": "05 / Contact",
      "[data-section='contact'] h2": "Shall we build<br /><em>something smart?</em>",
      "[data-section='contact'] .contact-inner > p:not(.eyebrow)": "I’m open to collaborations, opportunities and conversations about data and Artificial Intelligence.",
      "[data-section='contact'] .button-primary": 'Send me an email <span aria-hidden="true">↗</span>',
      ".skills-list": "<span>01 &middot; Python &amp; SQL</span><span>02 &middot; Predictive modelling</span><span>03 &middot; NLP</span><span>04 &middot; Data visualisation</span>",
      "[data-project='housing'] .project-label": "REGRESSION &middot; XGBOOST",
      "[data-project='fraud'] .project-label": "CLASSIFICATION &middot; SHAP",
      "[data-project='housing'] .art-caption": "California Housing &middot; test R²",
      "[data-project='text'] .art-caption": "20 Newsgroups &middot; macro F1 (20 classes)",
      "[data-project='fraud'] .art-caption": "Credit Card Fraud &middot; 0.17 % positives",
      "[data-project='fraud'] .fraud-stat small": "test precision",
      ".contact-copy": "Copy",
      ".footer-linkedin": "LinkedIn ↗",
      ".footer a:not(.footer-linkedin)": "Back to top ↑",
    },
    methods: [
      ["The test set is touched once", "A 60/20/20 split. Model, hyperparameters and threshold are all chosen by looking at <b>validation</b>. The test set is used at the end, with the winner, and takes part in no decision."],
      ["Every number with its spread", "A single figure from a single split says nothing. I publish cross-validation with its standard deviation: <b>R² 0.847 ± 0.007</b> is more useful, and more honest, than a lucky 0.91."],
      ["The right metric for the problem", "With 0.17% fraud, ROC-AUC looks good almost always and misleads. That is why project 03 is judged on <b>PR-AUC</b>, and project 02 on <b>macro F1</b> rather than accuracy."],
    ],
    metrics: {
      housing: [["R² test", "0.850"], ["R² cross-val", "0.847 ± 0.007"], ["MAE", "≈ $29,600"]],
      text: [["Macro F1", "0.706"], ["Accuracy", "0.719"], ["Chance", "0.050"]],
      fraud: [["Precision", "0.931"], ["Recall", "0.818"], ["PR-AUC", "0.874"]],
    },
  },
};

/* --- Links on each project card ------------------------------------- */
function renderLinks(t) {
  Object.entries(CONFIG.projects).forEach(([key, { folder, demo }]) => {
    const container = document.querySelector(`[data-project="${key}"] .project-links`);
    if (!container) return;
    const repo = `${CONFIG.github}/tree/main/${folder}`;
    // With no deployed demo we avoid a dead link and point at the README.
    const primary = demo
      ? `<a href="${demo}" target="_blank" rel="noreferrer">${t.demo}</a>`
      : `<a class="is-local" href="${repo}#run" target="_blank" rel="noreferrer">${t.local}</a>`;
    container.innerHTML =
      `${primary}<a href="${repo}" target="_blank" rel="noreferrer">${t.code}</a>`;
  });
}

/* --- Metric lists on each project card ------------------------------ */
function renderMetrics(t) {
  Object.entries(t.metrics).forEach(([key, rows]) => {
    const list = document.querySelector(`[data-project="${key}"] .metric-list`);
    if (list) {
      list.innerHTML = rows.map(([k, v]) => `<li><b>${k}</b><span>${v}</span></li>`).join("");
    }
  });
}

// Labels for the copy button in the active language; setLanguage() keeps
// them in sync, so this must be initialised before the first call.
let copyLabels = { idle: "Copy", done: "Copied" };

/* --- Language switching ---------------------------------------------- */
function setLanguage(language) {
  const t = TEXTS[language] || TEXTS.en;

  const navLinks = document.querySelectorAll(".nav > a:not(.nav-cta), .nav-cta");
  t.nav.forEach((label, i) => {
    const a = navLinks[i];
    if (!a) return;
    // The CTA carries an arrow in a <span>; only the text node is replaced.
    if (a.firstChild) a.firstChild.textContent = a.classList.contains("nav-cta") ? `${label} ` : label;
  });

  Object.entries(t.blocks).forEach(([selector, html]) => {
    const el = document.querySelector(selector);
    if (el) el.innerHTML = html;
  });

  document.querySelectorAll(".method-card").forEach((card, i) => {
    const content = t.methods[i];
    if (!content) return;
    card.querySelector("h3").textContent = content[0];
    card.querySelector("p").innerHTML = content[1];
  });

  renderMetrics(t);
  renderLinks(t);
  copyLabels = { idle: t.blocks[".contact-copy"], done: t.copied };

  document.documentElement.lang = language;
  try {
    localStorage.setItem("portfolio-language", language);
  } catch (e) {
    /* private mode: we carry on without remembering the language */
  }
  document.querySelectorAll(".language-button").forEach((button) => {
    const active = button.dataset.language === language;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

document.querySelectorAll(".language-button").forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.language));
});

// English is the default; the visitor's previous choice wins over it.
let saved = "en";
try {
  saved = localStorage.getItem("portfolio-language") || "en";
} catch (e) {
  /* localStorage unavailable */
}
setLanguage(saved);

/* --- Copy the contact address ----------------------------------------
   mailto: does nothing on a machine with no mail client configured, so the
   address is also offered as one-click copy. The execCommand branch covers
   browsers that block the async clipboard API outside a secure context. */
document.querySelectorAll(".contact-copy").forEach((button) => {
  button.addEventListener("click", async () => {
    const value = button.dataset.copy;
    let ok = true;
    try {
      await navigator.clipboard.writeText(value);
    } catch (e) {
      const field = document.createElement("textarea");
      field.value = value;
      field.setAttribute("readonly", "");
      field.style.position = "fixed";
      field.style.opacity = "0";
      document.body.appendChild(field);
      field.select();
      try {
        ok = document.execCommand("copy");
      } catch (err) {
        ok = false;
      }
      field.remove();
    }
    if (!ok) return;
    button.textContent = copyLabels.done;
    button.classList.add("copied");
    setTimeout(() => {
      button.textContent = copyLabels.idle;
      button.classList.remove("copied");
    }, 2000);
  });
});

/* --- Mobile menu ------------------------------------------------------ */
const menuToggle = document.querySelector(".menu-toggle");
const nav = document.querySelector(".nav");

menuToggle?.addEventListener("click", () => {
  const open = nav.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(open));
});

document.querySelectorAll(".nav a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("open");
    menuToggle?.setAttribute("aria-expanded", "false");
  });
});

/* --- Progressive reveal on scroll ------------------------------------ */
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
