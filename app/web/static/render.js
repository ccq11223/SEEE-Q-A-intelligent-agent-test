/* AI 回答渲染：Markdown + LaTeX（KaTeX）+ DOMPurify 安全过滤。
 *
 * 先把公式摘出来换成占位符，再做 Markdown 解析和消毒，最后把占位符换回
 * KaTeX 渲染结果。这样能避免 Markdown 把 LaTeX 里的下划线、星号当成
 * 强调符号吃掉，是「公式 + Markdown 混排」最稳妥的做法。
 */
(function () {
  'use strict';

  var MATH_RE = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^$\n]+?\$|\\\([^\\\n]+?\\\))/g;
  var PLACEHOLDER_RE = /\u0000MATH(\d+)\u0000/g;

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
      return '\u0000MATH' + (math.length - 1) + '\u0000';
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
})();
