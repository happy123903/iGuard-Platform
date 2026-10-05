(function () {
  const savedTheme = localStorage.getItem("iguard-theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);

  window.toggleTheme = function () {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("iguard-theme", newTheme);
    updateThemeUI(newTheme);

    // Dispatch event for any interested components (e.g. Chart.js in admin)
    window.dispatchEvent(new CustomEvent("themeChanged", { detail: { theme: newTheme } }));
  };

  function updateThemeUI(theme) {
    const textElem = document.getElementById("theme-text");
    const btnElem = document.getElementById("theme-toggle-btn");
    if (textElem) {
      textElem.textContent = theme === "dark" ? "夜間模式" : "白天模式";
    }
    if (btnElem) {
      btnElem.setAttribute(
        "title",
        theme === "dark" ? "目前為夜間模式，點擊切換為白天模式" : "目前為白天模式，點擊切換為夜間模式"
      );
      btnElem.setAttribute("aria-label", theme === "dark" ? "切換至白天模式" : "切換至夜間模式");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      updateThemeUI(document.documentElement.getAttribute("data-theme") || "light");
    });
  } else {
    updateThemeUI(savedTheme);
  }
})();
