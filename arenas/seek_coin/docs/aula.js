/* Navegação das aulas em slides: setas, PageUp/PageDown, Home/End,
   botões e barra de progresso. Compartilhado pelas docs HTML. */
(() => {
  const slides = [...document.querySelectorAll(".slide")];
  if (!slides.length) return;
  let current = 0;

  const fill = document.getElementById("progress-fill");
  const counter = document.getElementById("counter");
  const section = document.getElementById("sec-label");
  const pad = (n) => String(n).padStart(2, "0");

  function show(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      slide.hidden = i !== current;
      slide.classList.toggle("active", i === current);
    });
    if (fill) fill.style.width = `${((current + 1) / slides.length) * 100}%`;
    if (counter) counter.textContent = `${pad(current + 1)} / ${pad(slides.length)}`;
    if (section) section.textContent = slides[current].dataset.section || "";
    slides[current].scrollTop = 0;
    if (slides[current].id) history.replaceState(null, "", `#${slides[current].id}`);
  }

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (["ArrowRight", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      show(current + 1);
    } else if (["ArrowLeft", "PageUp"].includes(event.key)) {
      event.preventDefault();
      show(current - 1);
    } else if (event.key === "Home") {
      show(0);
    } else if (event.key === "End") {
      show(slides.length - 1);
    }
  });

  document.getElementById("btn-next")?.addEventListener("click", () => show(current + 1));
  document.getElementById("btn-prev")?.addEventListener("click", () => show(current - 1));

  const fromHash = slides.findIndex((slide) => `#${slide.id}` === location.hash);
  show(fromHash >= 0 ? fromHash : 0);
})();
