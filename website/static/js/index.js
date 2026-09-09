"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const scrollButton = document.querySelector(".scroll-to-top");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const updateScrollButton = () => {
    scrollButton?.classList.toggle("visible", window.scrollY > 300);
  };
  scrollButton?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: reducedMotion.matches ? "auto" : "smooth" });
  });
  window.addEventListener("scroll", updateScrollButton, { passive: true });
  updateScrollButton();

  // Native video controls remain usable when autoplay is unavailable.
  if (reducedMotion.matches) {
    document.querySelectorAll("video[autoplay]").forEach((video) => {
      video.removeAttribute("autoplay");
      video.pause();
    });
  }
});
