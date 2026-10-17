/* AI 回答渲染：Markdown + LaTeX（KaTeX）+ DOMPurify 安全过滤。
 *
 * 先把公式摘出来换成占位符，再做 Markdown 解析和消毒，最后把占位符换回
 * KaTeX 渲染结果。这样能避免 Markdown 把 LaTeX 里的下划线、星号当成
 * 强调符号吃掉，是「公式 + Markdown 混排」最稳妥的做法。
 */
(function () {
  'use strict';

  var MATH_RE = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^$\n]+?\$|\\\([^\\\n]+?\\\))/g;
  // 占位符必须是普通可见 ASCII：早先用 \u0000 包裹，经过 DOMPurify 的 HTML
  // 解析时控制字符会被替换/丢弃，最后页面上就只剩下 “MATH3” 这种文字。
  var PLACEHOLDER_RE = /%%MATH(\d+)%%/g;

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderTex(source) {
    var tex = source;
    var display = false;
    if (/^\$\$/.test(tex) || /^\\\[/.test(tex)) {
      display = true;
      tex = tex.replace(/^\$\$|\$\$$/g, '').replace(/^\\\[|\\\]$/g, '');
    } else {
      tex = tex.replace(/^\$|\$$/g, '').replace(/^\\\(|\\\)$/g, '');
    }
    if (!tex.trim()) return '';
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false });
    } catch (err) {
      return '<code>' + escapeHtml(source) + '</code>';
    }
  }

  window.renderRich = function (raw) {
    var math = [];
    var text = String(raw == null ? '' : raw).replace(MATH_RE, function (match) {
      math.push(match);
      return '%%MATH' + (math.length - 1) + '%%';
    });

    var html;
    try {
      html = marked.parse(text, { breaks: true, gfm: true });
    } catch (err) {
      html = '<p>' + escapeHtml(text) + '</p>';
    }

    html = DOMPurify.sanitize(html);
    return html.replace(PLACEHOLDER_RE, function (_, index) {
      return renderTex(math[Number(index)]);
    });
  };

  function extractPageNumbers(text) {
    var pages = {};
    var block = /第\s*(\d{1,4}(?:\s*[、,，/~\-—–−至]\s*\d{1,4})*)\s*页/g;
    var match;
    while ((match = block.exec(text || ''))) {
      var numbers = match[1].match(/\d{1,4}/g) || [];
      numbers.forEach(function (n) { pages[Number(n)] = true; });
    }
    return pages;
  }

  function buildFigure(image) {
    var fig = document.createElement('figure');
    fig.className = 'fig';
    var pic = document.createElement('img');
    pic.src = image.url;
    pic.alt = image.label || '课件原页';
    pic.loading = 'lazy';
    var cap = document.createElement('figcaption');
    cap.textContent = (image.label || '课件原页') + ' · 课件原图';
    fig.appendChild(pic);
    fig.appendChild(cap);
    return fig;
  }

  /* 把课件原图插到「提到这一页」的段落后面，而不是在回答末尾堆成一排。
     这样图片跟着对应讲解走，读起来像老师边讲边指图。 */
  window.insertPageImages = function (container, images) {
    var pending = images.slice();
    var used = {};
    var blocks = Array.prototype.slice
      .call(container.querySelectorAll('p, li, blockquote'))
      .filter(function (el) { return !el.querySelector('p, li, blockquote'); });

    var candidates = blocks.map(function (block) {
      return {
        el: block,
        pages: extractPageNumbers(block.textContent || ''),
        used: false
      };
    });

    function place(candidate, image) {
      var fig = buildFigure(image);
      if (candidate.el.tagName === 'LI') {
        candidate.el.appendChild(fig);
      } else {
        candidate.el.insertAdjacentElement('afterend', fig);
      }
      candidate.used = true;
      used[image.url] = true;
    }

    // 每段最多配一张图：先给「提到了这一页、且还没有配图」的段落，
    // 这样多张图会自然分散到各自的讲解段落，而不是全堆在第一段后面。
    pending.forEach(function (image) {
      for (var i = 0; i < candidates.length; i += 1) {
        var candidate = candidates[i];
        if (candidate.used) continue;
        if (!candidate.pages[Number(image.page)]) continue;
        place(candidate, image);
        return;
      }
    });

    // 段落不够分时，再允许同一段配第二张
    pending.forEach(function (image) {
      if (used[image.url]) return;
      for (var i = 0; i < candidates.length; i += 1) {
        if (!candidates[i].pages[Number(image.page)]) continue;
        place(candidates[i], image);
        return;
      }
    });

    // 模型没在正文里点到页码时，至少把图放在最后，不要丢
    var rest = pending.filter(function (image) { return !used[image.url]; });
    if (!rest.length) return;
    var figs = document.createElement('div');
    figs.className = 'figs';
    rest.forEach(function (image) { figs.appendChild(buildFigure(image)); });
    container.appendChild(figs);
  };
})();
