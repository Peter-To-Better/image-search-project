// 與 FastAPI 後端溝通的小工具
async function handle(res) {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export const getInfo = () => fetch("/api/info").then(handle);
export const getSamples = (kind = "semantic", n = 12, seed) =>
  fetch(`/api/samples?kind=${kind}&n=${n}${seed != null ? `&seed=${seed}` : ""}`).then(handle);
export const getResults = () => fetch("/api/results").then(handle);

// query = { type: 'file', file } | { type: 'gallery', gallery_id } | { type: 'query', path }
function queryForm(query, extra) {
  const fd = new FormData();
  if (query.type === "file") fd.append("file", query.file);
  if (query.type === "gallery") fd.append("gallery_id", query.gallery_id);
  if (query.type === "query") fd.append("query_path", query.path);
  Object.entries(extra).forEach(([k, v]) => v != null && fd.append(k, v));
  return fd;
}

export const search = (query, opts) =>
  fetch("/api/search", { method: "POST", body: queryForm(query, opts) }).then(handle);

export const compare = (query, opts) =>
  fetch("/api/compare", { method: "POST", body: queryForm(query, opts) }).then(handle);

export const textSearch = (text, index = "brute", k = 12) => {
  const fd = new FormData();
  fd.append("text", text);
  fd.append("index", index);
  fd.append("k", k);
  return fetch("/api/text", { method: "POST", body: fd }).then(handle);
};

export const imgUrl = (path) => "/" + path;
