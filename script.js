/* ============================================================
   CONFIGURACIÓN — lo único que necesitas editar al desplegar
   ============================================================
   1. Sustituye TU-USUARIO por tu usuario real de GitHub.
   2. Cuando despliegues cada demo (Streamlit Community Cloud o
      Hugging Face Spaces), pega su URL aquí.
      Mientras una demo esté vacía, la tarjeta muestra "Ejecutar
      en local" y enlaza al README en vez de a un enlace muerto.
   ============================================================ */
const CONFIG = {
  github: "https://github.com/TU-USUARIO/portafolio-ia-ml",
  proyectos: {
    viviendas: { carpeta: "01_prediccion_viviendas", demo: "" },
    texto:     { carpeta: "02_clasificador_texto",   demo: "" },
    fraude:    { carpeta: "03_deteccion_fraude",     demo: "" },
  },
};

const TEXTOS = {
  es: {
    nav: ["Proyectos", "Metodología", "Sobre mí", "Hablemos"],
    demo: 'Abrir demo <span aria-hidden="true">↗</span>',
    local: 'Ejecutar en local <span aria-hidden="true">↗</span>',
    codigo: 'Ver código <span aria-hidden="true">↗</span>',
    bloques: {
      "[data-section='hero'] .eyebrow": '<span class="status-dot"></span> Disponible para proyectos de IA',
      "[data-section='hero'] h1": "Convierto datos<br /><em>en decisiones.</em>",
      "[data-section='hero'] .hero-lede": "Soy <strong>Fernando Tafurt Pinto</strong>, especialista en formación en Inteligencia Artificial y Machine Learning. Construyo modelos que resuelven problemas reales y publico sus métricas sin maquillar.",
      "[data-section='hero'] .button-primary": 'Ver mis proyectos <span aria-hidden="true">↓</span>',
      "[data-section='hero'] .button-ghost": 'Conocerme mejor <span aria-hidden="true">↗</span>',
      "[data-section='projects'] .eyebrow": "01 / Proyectos seleccionados",
      "[data-section='projects'] h2": "Problemas reales.<br /><em>Métricas honestas.</em>",
      "[data-section='projects'] .section-note": "Tres familias distintas de problema: regresión tabular, clasificación de texto y clasificación con clases desbalanceadas. Cada uno con su código, sus métricas y sus limitaciones documentadas.",
      "[data-project='viviendas'] h3": "Predicción de precios de vivienda",
      "[data-project='viviendas'] .project-body > p": "Regresión sobre 20.640 distritos censales de California. Comparación de tres modelos y demo interactiva.",
      "[data-project='texto'] h3": "Clasificador de texto por tema",
      "[data-project='texto'] .project-body > p": "Clasificación de mensajes en 20 categorías. Explica qué términos concretos empujaron cada decisión.",
      "[data-project='fraude'] h3": "Detección de fraude",
      "[data-project='fraude'] .project-body > p": "Clases muy desbalanceadas sobre 284.807 transacciones reales. Umbral elegido en validación y explicación con SHAP.",
      "[data-section='method'] .eyebrow": "02 / Cómo trabajo",
      "[data-section='method'] h2": "Una métrica solo vale<br /><em>si es honesta.</em>",
      "[data-section='method'] .section-note": "Es fácil inflar un resultado sin darse cuenta. Estas son las tres reglas que sigo en los tres proyectos, y son la razón de que mis cifras sean más bajas de lo que podrían parecer.",
      "[data-section='about'] .eyebrow": "03 / Sobre mí",
      "[data-section='about'] h2": "La curiosidad es<br /><em>mi combustible.</em>",
      "[data-section='about'] .about-copy > p:not(.eyebrow)": "Me interesa entender cómo los datos pueden ayudarnos a tomar mejores decisiones. Mi trabajo combina fundamentos sólidos de Machine Learning, comunicación clara y el deseo de construir soluciones que generen impacto.",
      "[data-section='contact'] .eyebrow": "04 / Contacto",
      "[data-section='contact'] h2": "¿Construimos algo<br /><em>inteligente?</em>",
      "[data-section='contact'] .contact-inner > p:not(.eyebrow)": "Estoy abierto a colaboraciones, oportunidades y conversaciones sobre datos e Inteligencia Artificial.",
      "[data-section='contact'] .button": 'Escríbeme un correo <span aria-hidden="true">↗</span>',
      ".certificate-note p": "<strong>Formación y certificaciones</strong><br />Esta sección incorporará mis credenciales con su emisor, fecha y enlace de verificación.",
      ".certificate-note .text-link": "Hablemos <span>↗</span>",
      ".footer a": "Volver arriba ↑",
    },
    metodos: [
      ["El test se toca una sola vez", "Partición 60/20/20. El modelo, los hiperparámetros y el umbral se eligen mirando <b>validación</b>. El test se usa al final, con el ganador, y no interviene en ninguna decisión."],
      ["Cada número con su desviación", "Una cifra suelta de una sola partición no dice nada. Publico validación cruzada con su desviación: <b>R² 0.847 ± 0.007</b> es más útil, y más honesto, que un 0.91 afortunado."],
      ["La métrica correcta para el problema", "Con un 0,17 % de fraude, ROC-AUC se ve bien casi siempre y engaña. Por eso el proyecto 03 se juzga con <b>PR-AUC</b>, y el 02 con <b>F1 macro</b> en lugar de exactitud."],
    ],
    metricas: {
      viviendas: [["R² test", "0.850"], ["R² val. cruzada", "0.847 ± 0.007"], ["MAE", "≈ $29.600"]],
      texto: [["F1 macro", "0.706"], ["Exactitud", "0.719"], ["Azar", "0.050"]],
      fraude: [["Precisión", "0.931"], ["Recall", "0.818"], ["PR-AUC", "0.874"]],
    },
  },

  en: {
    nav: ["Projects", "Methodology", "About me", "Let’s talk"],
    demo: 'Open demo <span aria-hidden="true">↗</span>',
    local: 'Run locally <span aria-hidden="true">↗</span>',
    codigo: 'View code <span aria-hidden="true">↗</span>',
    bloques: {
      "[data-section='hero'] .eyebrow": '<span class="status-dot"></span> Available for AI projects',
      "[data-section='hero'] h1": "I turn data<br /><em>into decisions.</em>",
      "[data-section='hero'] .hero-lede": "I’m <strong>Fernando Tafurt Pinto</strong>, an AI and Machine Learning professional in training. I build models that solve real problems and I publish their metrics unpolished.",
      "[data-section='hero'] .button-primary": 'View my projects <span aria-hidden="true">↓</span>',
      "[data-section='hero'] .button-ghost": 'Get to know me <span aria-hidden="true">↗</span>',
      "[data-section='projects'] .eyebrow": "01 / Selected projects",
      "[data-section='projects'] h2": "Real problems.<br /><em>Honest metrics.</em>",
      "[data-section='projects'] .section-note": "Three different families of problem: tabular regression, text classification and classification with imbalanced classes. Each one with its code, its metrics and its limitations documented.",
      "[data-project='viviendas'] h3": "Housing price prediction",
      "[data-project='viviendas'] .project-body > p": "Regression over 20,640 California census districts. Three models compared plus an interactive demo.",
      "[data-project='texto'] h3": "Topic classifier for text",
      "[data-project='texto'] .project-body > p": "Message classification across 20 categories. It shows which specific terms drove each decision.",
      "[data-project='fraude'] h3": "Fraud detection",
      "[data-project='fraude'] .project-body > p": "Heavily imbalanced classes over 284,807 real transactions. Threshold picked on validation and explained with SHAP.",
      "[data-section='method'] .eyebrow": "02 / How I work",
      "[data-section='method'] h2": "A metric only counts<br /><em>if it is honest.</em>",
      "[data-section='method'] .section-note": "It is easy to inflate a result without noticing. These are the three rules I follow across all three projects, and the reason my numbers are lower than they could look.",
      "[data-section='about'] .eyebrow": "03 / About me",
      "[data-section='about'] h2": "Curiosity is<br /><em>my fuel.</em>",
      "[data-section='about'] .about-copy > p:not(.eyebrow)": "I’m interested in understanding how data can help us make better decisions. My work combines strong Machine Learning foundations, clear communication and a desire to build solutions that create impact.",
      "[data-section='contact'] .eyebrow": "04 / Contact",
      "[data-section='contact'] h2": "Shall we build<br /><em>something smart?</em>",
      "[data-section='contact'] .contact-inner > p:not(.eyebrow)": "I’m open to collaborations, opportunities and conversations about data and Artificial Intelligence.",
      "[data-section='contact'] .button": 'Send me an email <span aria-hidden="true">↗</span>',
      ".certificate-note p": "<strong>Education and certifications</strong><br />This section will list my credentials with their issuer, date and verification link.",
      ".certificate-note .text-link": "Let’s talk <span>↗</span>",
      ".footer a": "Back to top ↑",
    },
    metodos: [
      ["The test set is touched once", "A 60/20/20 split. Model, hyperparameters and threshold are all chosen by looking at <b>validation</b>. The test set is used at the end, with the winner, and takes part in no decision."],
      ["Every number with its spread", "A single figure from a single split says nothing. I publish cross-validation with its standard deviation: <b>R² 0.847 ± 0.007</b> is more useful, and more honest, than a lucky 0.91."],
      ["The right metric for the problem", "With 0.17% fraud, ROC-AUC looks good almost always and misleads. That is why project 03 is judged on <b>PR-AUC</b>, and project 02 on <b>macro F1</b> rather than accuracy."],
    ],
    metricas: {
      viviendas: [["R² test", "0.850"], ["R² cross-val", "0.847 ± 0.007"], ["MAE", "≈ $29,600"]],
      texto: [["Macro F1", "0.706"], ["Accuracy", "0.719"], ["Chance", "0.050"]],
      fraude: [["Precision", "0.931"], ["Recall", "0.818"], ["PR-AUC", "0.874"]],
    },
  },
};

/* --- Enlaces de cada proyecto -------------------------------------- */
function pintarEnlaces(t) {
  Object.entries(CONFIG.proyectos).forEach(([clave, { carpeta, demo }]) => {
    const contenedor = document.querySelector(`[data-project="${clave}"] .project-links`);
    if (!contenedor) return;
    const repo = `${CONFIG.github}/tree/main/${carpeta}`;
    // Sin demo desplegada no ponemos un enlace muerto: mandamos al README.
    const principal = demo
      ? `<a href="${demo}" target="_blank" rel="noreferrer">${t.demo}</a>`
      : `<a class="is-local" href="${repo}#ejecutar" target="_blank" rel="noreferrer">${t.local}</a>`;
    contenedor.innerHTML =
      `${principal}<a href="${repo}" target="_blank" rel="noreferrer">${t.codigo}</a>`;
  });
}

/* --- Listas de métricas de cada tarjeta ----------------------------- */
function pintarMetricas(t) {
  Object.entries(t.metricas).forEach(([clave, filas]) => {
    const lista = document.querySelector(`[data-project="${clave}"] .metric-list`);
    if (lista) {
      lista.innerHTML = filas.map(([k, v]) => `<li><b>${k}</b><span>${v}</span></li>`).join("");
    }
  });
}

/* --- Cambio de idioma ----------------------------------------------- */
function setLanguage(idioma) {
  const t = TEXTOS[idioma] || TEXTOS.es;

  const enlacesNav = document.querySelectorAll(".nav > a:not(.nav-cta), .nav-cta");
  t.nav.forEach((etiqueta, i) => {
    const a = enlacesNav[i];
    if (!a) return;
    // El CTA lleva una flecha en un <span>; solo cambiamos el nodo de texto.
    if (a.firstChild) a.firstChild.textContent = a.classList.contains("nav-cta") ? `${etiqueta} ` : etiqueta;
  });

  Object.entries(t.bloques).forEach(([selector, html]) => {
    const el = document.querySelector(selector);
    if (el) el.innerHTML = html;
  });

  document.querySelectorAll(".method-card").forEach((card, i) => {
    const contenido = t.metodos[i];
    if (!contenido) return;
    card.querySelector("h3").textContent = contenido[0];
    card.querySelector("p").innerHTML = contenido[1];
  });

  pintarMetricas(t);
  pintarEnlaces(t);

  document.documentElement.lang = idioma;
  try {
    localStorage.setItem("portfolio-language", idioma);
  } catch (e) {
    /* modo privado: seguimos sin recordar el idioma */
  }
  document.querySelectorAll(".language-button").forEach((boton) => {
    const activo = boton.dataset.language === idioma;
    boton.classList.toggle("active", activo);
    boton.setAttribute("aria-pressed", String(activo));
  });
}

document.querySelectorAll(".language-button").forEach((boton) => {
  boton.addEventListener("click", () => setLanguage(boton.dataset.language));
});

let guardado = "es";
try {
  guardado = localStorage.getItem("portfolio-language") || "es";
} catch (e) {
  /* sin localStorage disponible */
}
setLanguage(guardado);

/* --- Menú móvil ------------------------------------------------------ */
const menuToggle = document.querySelector(".menu-toggle");
const nav = document.querySelector(".nav");

menuToggle?.addEventListener("click", () => {
  const abierto = nav.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(abierto));
});

document.querySelectorAll(".nav a").forEach((enlace) => {
  enlace.addEventListener("click", () => {
    nav.classList.remove("open");
    menuToggle?.setAttribute("aria-expanded", "false");
  });
});

/* --- Aparición progresiva al hacer scroll ---------------------------- */
const observer = new IntersectionObserver(
  (entradas) => {
    entradas.forEach((entrada) => {
      if (entrada.isIntersecting) {
        entrada.target.classList.add("is-visible");
        observer.unobserve(entrada.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
