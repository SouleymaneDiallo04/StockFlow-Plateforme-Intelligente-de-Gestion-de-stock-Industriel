/* ============================================================
   StockFlow Intelligence — main.js
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Theme toggle ───────────────────────────────────────────
  const savedTheme = localStorage.getItem('sf-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  document.querySelectorAll('.sf-theme-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('sf-theme', next);
      updateThemeIcon(next);
    });
  });

  function updateThemeIcon(theme) {
    document.querySelectorAll('.sf-theme-icon').forEach(el => {
      el.textContent = theme === 'dark' ? '☀️' : '🌙';
    });
  }

  // ── Toasts depuis messages Django ─────────────────────────
  const msgEl = document.getElementById('django-messages');
  if (msgEl) {
    try {
      const msgs = JSON.parse(msgEl.textContent);
      msgs.forEach(({ message, tags }) => createToast(message, tags || 'info'));
    } catch(e) {}
  }

  function createToast(message, tag) {
    const icons = { success: '✓', danger: '✕', warning: '⚠', info: 'ℹ' };
    const container = document.getElementById('sf-toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `sf-toast sf-toast-${tag}`;
    toast.innerHTML = `<span class="sf-toast-icon">${icons[tag] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('hiding');
      toast.addEventListener('animationend', () => toast.remove());
    }, 4500);
  }

  // ── Sidebar mobile ─────────────────────────────────────────
  const sidebar = document.querySelector('.sf-sidebar');
  const overlay = document.getElementById('sf-overlay');

  document.querySelectorAll('.sf-menu-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      sidebar?.classList.toggle('open');
      overlay?.classList.toggle('show');
    });
  });

  overlay?.addEventListener('click', () => {
    sidebar?.classList.remove('open');
    overlay.classList.remove('show');
  });

  // ── Counter animation ──────────────────────────────────────
  function animateCounter(el) {
    const target = parseInt(el.textContent.replace(/[^\d]/g, ''), 10);
    if (isNaN(target) || target === 0) return;
    const duration = 900;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(eased * target);
      if (progress < 1) requestAnimationFrame(step);
    };
    el.textContent = '0';
    requestAnimationFrame(step);
  }

  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.sf-counter').forEach(el => counterObserver.observe(el));

  // ── Image preview ──────────────────────────────────────────
  const photoInput = document.getElementById('photoInput');
  const imgPreview = document.getElementById('imgPreview');
  if (photoInput && imgPreview) {
    photoInput.addEventListener('change', () => {
      const file = photoInput.files[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = e => {
          imgPreview.src = e.target.result;
          imgPreview.classList.add('show');
        };
        reader.readAsDataURL(file);
      } else {
        imgPreview.classList.remove('show');
      }
    });
  }

  // ── Autocomplete produits ──────────────────────────────────
  const searchInput     = document.getElementById('sf-search-input');
  const autocompleteBox = document.getElementById('sf-autocomplete');

  if (searchInput && autocompleteBox) {
    let debounceTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      const q = searchInput.value.trim();
      if (q.length < 2) {
        autocompleteBox.innerHTML = '';
        autocompleteBox.hidden = true;
        return;
      }
      debounceTimer = setTimeout(async () => {
        try {
          const res  = await fetch(`/products/autocomplete/?q=${encodeURIComponent(q)}`);
          const data = await res.json();
          if (!data.results.length) { autocompleteBox.hidden = true; return; }
          autocompleteBox.innerHTML = data.results.map(p =>
            `<a class="sf-autocomplete-item" href="/products/${p.pk}/">
               ${p.name}<span>${p.reference}</span>
             </a>`
          ).join('');
          autocompleteBox.hidden = false;
        } catch(e) {}
      }, 220);
    });

    document.addEventListener('click', e => {
      if (!searchInput.contains(e.target) && !autocompleteBox.contains(e.target)) {
        autocompleteBox.hidden = true;
      }
    });
  }

  // ── Confirm delete double-check ────────────────────────────
  const confirmCheckbox = document.getElementById('confirm-delete-check');
  const confirmBtn      = document.getElementById('confirm-delete-btn');
  if (confirmCheckbox && confirmBtn) {
    confirmBtn.disabled = true;
    confirmCheckbox.addEventListener('change', () => {
      confirmBtn.disabled = !confirmCheckbox.checked;
    });
  }

  // ── Active nav link ────────────────────────────────────────
  const currentPath = window.location.pathname;
  document.querySelectorAll('.sf-nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (!href) return;
    if (href === '/' && currentPath === '/') {
      link.classList.add('active');
    } else if (href !== '/' && currentPath.startsWith(href)) {
      link.classList.add('active');
    }
  });

  // ── HTMX : après swap, relancer les compteurs ──────────────
  document.body.addEventListener('htmx:afterSwap', () => {
    document.querySelectorAll('.sf-counter').forEach(el => {
      if (!el.dataset.animated) {
        el.dataset.animated = '1';
        animateCounter(el);
      }
    });
  });

});
