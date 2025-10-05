import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import ForceGraph2D from "react-force-graph-2d";
import type { ForceGraphMethods, NodeObject } from "react-force-graph-2d";
import { Search, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

// 🌐 CONFIGURACIÓN DEL BACKEND
const API_URL = "http://127.0.0.1:8000/api/query_json";

// 🔧 TIPOS
type Articulo = { titulo: string; link: string };
type Nodo = {
  palabra: string;
  articulos: Articulo[];
  relaciones: string[];
};

type UINode = NodeObject & {
  label?: string;
  x?: number; y?: number;
  fx?: number; fy?: number;
};

type DataJSON = {
  reporte: { resumen: string; hallazgos: string[] };
  grafo: Nodo[];
};

type GraphNode = {
  id: string;
  label: string;
  articulos: Articulo[];
};

type GraphLink = { source: string; target: string };

type GraphData = { nodes: GraphNode[]; links: GraphLink[] };

// 📦 Datos por defecto (fallback) en caso de fallo de la API o inicialización
const defaultData: DataJSON = {
  reporte: {
    resumen:
      "Error: No se pudo cargar el reporte del backend. Mostrando datos de ejemplo. Por favor, asegúrate de que el servidor FastAPI esté corriendo en la URL configurada.",
    hallazgos: [
      "Verifica que el servidor Uvicorn esté activo en http://127.0.0.1:8000.",
      "Revisa la consola del servidor de FastAPI en busca de errores críticos de inicialización.",
      "Confirma que la URL en fetchRAGReport (API_URL) es correcta.",
      "Si ves errores CORS en la consola del navegador, configura el CORS en tu servidor FastAPI."
    ],
  },
  grafo: [
    {
      palabra: "IA generativa",
      articulos: [
        {
          titulo: "IA generativa para fomentar la creatividad en el aula: un estudio exploratorio",
          link: "https://github.com/jgalazka/SB_publications/tree/main/education/2024_ia_generativa_creatividad_aula.pdf",
        },
      ],
      relaciones: ["creatividad", "evaluación", "pensamiento de diseño"],
    },
    {
      palabra: "creatividad",
      articulos: [
        {
          titulo: "Medición de creatividad asistida por IA: rúbricas y casos",
          link: "https://github.com/jgalazka/SB_publications/tree/main/assessment/2023_medicion_creatividad_rubricas.pdf",
        },
      ],
      relaciones: ["IA generativa", "pensamiento de diseño"],
    },
  ],
};

// 🌐 FUNCIÓN DE LLAMADA HTTP AL BACKEND (FastAPI)
async function fetchRAGReport(question: string): Promise<DataJSON | null> {
  const dataToSend = {
    question: question,
    k_chunks: 5,
  };

  console.log(`Enviando consulta a la API: ${question}`);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(dataToSend),
    });

    if (!response.ok) {
      console.error(`Error HTTP ${response.status}: ${response.statusText}`);
      return null;
    }

    const resultJson = (await response.json()) as DataJSON;
    return resultJson;
  } catch (error) {
    console.error("Error de red o CORS al obtener el reporte:", error);
    return null;
  }
}

// 🧠 Utilidad: construir datos para el grafo a partir del JSON
const buildGraphData = (nodos: Nodo[]): GraphData => {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];

  const exists = new Set(nodos.map((n) => n.palabra));

  nodos.forEach((n) => {
    nodes.push({ id: n.palabra, label: n.palabra, articulos: n.articulos });
    n.relaciones.forEach((rel) => {
      if (exists.has(rel)) {
        links.push({ source: n.palabra, target: rel });
      }
    });
  });

  // Eliminar duplicados (A->B y B->A) quedándonos con una sola
  const seen = new Set<string>();
  const uniqueLinks = links.filter((l) => {
    const a = String(l.source);
    const b = String(l.target);
    const key = a < b ? `${a}__${b}` : `${b}__${a}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return { nodes, links: uniqueLinks };
};

// 🔎 Filtra el subgrafo centrado en el término buscado (nodo + vecinos)
const filterGraphByQuery = (data: DataJSON, query: string): { grafo: Nodo[] } => {
  const q = query.trim().toLowerCase();
  if (!q) return { grafo: data.grafo }; 

  const map = new Map<string, Nodo>();
  data.grafo.forEach((n) => map.set(n.palabra.toLowerCase(), n));
  
  const centerNodeKey = Array.from(map.keys()).find(key => key.includes(q));
  
  if (!centerNodeKey) return { grafo: data.grafo }; 

  const center = map.get(centerNodeKey)!;
  
  const finalNodes = new Set<Nodo>();
  finalNodes.add(center);
  
  center.relaciones.forEach((r) => {
      const neighbor = map.get(r.toLowerCase());
      if (neighbor) {
          finalNodes.add(neighbor);
      }
  });

  return { grafo: Array.from(finalNodes) };
};

// 🧩 Componentes UI
function Navbar({ onSearch }: { onSearch: (term: string) => void }) {
  const [term, setTerm] = useState("");

  return (
    <div className="sticky top-0 z-30 w-full border-b bg-white/80 backdrop-blur">
      <div className="mx-auto flex items-center justify-between gap-4 p-3 max-w-7xl">
        {/* Logo y nombre */}
        <div className="flex flex-shrink-0 items-center gap-3 min-w-[180px]">
          <img src="https://placehold.co/48x48/0f172a/ffffff?text=Bio" alt="Logo Bionova" className="h-12 w-12 rounded-lg" />
          <div className="text-3xl font-semibold tracking-tight whitespace-nowrap">
            Bionova
          </div>
        </div>

        {/* Barra de búsqueda */}
        <form
          className="flex flex-1 max-w-xl items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            onSearch(term);
          }}
        >
          <div className="relative flex-1">
            <Search
              className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-slate-400"
              size={18}
            />
            <Input
              className="pl-8 h-10 rounded-lg focus:ring-2 focus:ring-blue-500 transition duration-150"
              placeholder="Buscar tema (ej. Microgravedad, Microbioma, Órbita terrestre...)"
              value={term}
              onChange={(e) => setTerm(e.target.value)}
            />
          </div>
          <Button 
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 transition duration-150 text-white font-semibold rounded-lg shadow-md"
          >
            Buscar
          </Button>
        </form>
      </div>
    </div>
  );
}

function Reporte({ resumen, hallazgos }: { resumen: string; hallazgos: string[] }) {
  return (
    <Card className="shadow-xl border border-slate-200">
      <CardHeader>
        <div className="flex items-center gap-3">
          {/* Icono de Reporte */}
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-file-text text-blue-600"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>
          <CardTitle className="text-2xl font-bold text-slate-800">Reporte de Hallazgos RAG</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <p className="mb-5 leading-relaxed text-slate-700 p-3 bg-blue-50 border-l-4 border-blue-400 rounded-r-lg">
          {resumen}
        </p>
        <h3 className="text-lg font-semibold mt-4 mb-2 text-slate-800 border-b pb-1">Hallazgos Clave:</h3>
        <ul className="list-disc space-y-2 pl-6">
          {hallazgos.map((h, i) => (
            <li key={i} className="text-slate-800 text-sm">
              {h}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

// ⚠️ Componente Grafo simplificado (SVG estático) para asegurar la compilación
function Grafo({ data, onNodeClick }: { data: GraphData; onNodeClick: (n: GraphNode) => void }) {
  const fgRef = useRef<ForceGraphMethods | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  // Observa el tamaño del contenedor para fijar límites del área de dibujo
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        setDims({ w: Math.floor(cr.width), h: Math.floor(cr.height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Ajustar fuerza del grafo y auto-zoom al cargar
  useEffect(() => {
    if (!fgRef.current) return;
    const fg = fgRef.current;
    fg.d3Force("charge")?.strength(-220);
    // Zoom a contenido
    setTimeout(() => fg.zoomToFit(400, 60), 300);
  }, [data]);

  const PADDING = 20; // margen interno para que no toque los bordes

  // Limitar un nodo a los límites visibles (en coordenadas de pantalla)
  const clampNodeToBounds = (n: UINode) => {
    if (!fgRef.current) return;
    const { w, h } = dims;
    if (!w || !h) return;
    const scr = fgRef.current.graph2ScreenCoords(n.x ?? 0, n.y ?? 0);
    const sx = Math.min(w - PADDING, Math.max(PADDING, scr.x));
    const sy = Math.min(h - PADDING, Math.max(PADDING, scr.y));
    if (sx !== scr.x || sy !== scr.y) {
      const g = fgRef.current.screen2GraphCoords(sx, sy);
      n.x = g.x; n.y = g.y;
      if (n.fx != null && n.fy != null) { n.fx = g.x; n.fy = g.y; }
    }
  };

  const zoomIn = () => {
    if (!fgRef.current) return;
    const k = Math.min(4, (fgRef.current.zoom() || 1) * 1.2);
    fgRef.current.zoom(k, 250);
  };
  const zoomOut = () => {
    if (!fgRef.current) return;
    const k = Math.max(0.4, (fgRef.current.zoom() || 1) / 1.2);
    fgRef.current.zoom(k, 250);
  };
  const zoomFit = () => fgRef.current?.zoomToFit(400, 60);

  return (
    <Card className="shadow-md">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/icon_waypoints.svg" alt="Icono grafo" className="h-6 w-6" />
          <CardTitle className="text-xl">Grafo de conocimiento</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={zoomOut} title="Zoom out">−</Button>
          <Button size="sm" variant="outline" onClick={zoomIn} title="Zoom in">＋</Button>
          <Button size="sm" onClick={zoomFit} title="Ajustar a la vista">Ajustar</Button>
        </div>
      </CardHeader>
      <CardContent className="h-[560px]" ref={containerRef}>
        <ForceGraph2D
          ref={fgRef}
          graphData={data}
          width={dims.w || undefined}
          height={dims.h || undefined}
          minZoom={0.4}
          maxZoom={4}
          nodeCanvasObject={(node: UINode, ctx: CanvasRenderingContext2D, globalScale: number) => {
            // 📏 Legibilidad: tamaño de fuente estable y ocultar etiquetas si el zoom es muy pequeño
            const label = (node.label ?? "") as string;
            const hideLabels = globalScale > 3.0 ? false : globalScale < 0.35; // oculta si estás muy lejos
            const fontSize = hideLabels ? 0 : Math.max(11, 14 / Math.min(globalScale, 1.4));
            const padding = 6;

            // Nodo base (pastilla)
            if (!hideLabels) {
              ctx.font = `${fontSize}px Inter, system-ui, -apple-system`;
              const textWidth = ctx.measureText(label).width;
              const width = textWidth + padding * 2;
              const height = fontSize + padding * 1.2;
              ctx.fillStyle = "rgba(15,23,42,0.92)"; // slate-900
              drawRoundRect(ctx, node.x! - width / 2, node.y! - height / 2, width, height, 6);
              ctx.fill();

              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillStyle = "white";
              ctx.fillText(label, node.x!, node.y! + 1);
            } else {
              // Cuando las etiquetas están ocultas, dibuja un punto legible
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, 4, 0, 2 * Math.PI);
              ctx.fillStyle = "#0f172a"; // slate-900
              ctx.fill();
            }
          }}
          nodePointerAreaPaint={(node: UINode, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
            const label = (node.label ?? "") as string;
            const hideLabels = globalScale < 0.35;
            if (hideLabels) {
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, 8, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
              return;
            }
            const padding = 6;
            ctx.font = `12px Inter`;
            const textWidth = ctx.measureText(label).width;
            const width = textWidth + padding * 2;
            const height = 12 + padding * 1.2;
            ctx.fillStyle = color;
            ctx.fillRect(node.x! - width / 2, node.y! - height / 2, width, height);
          }}
          linkColor={() => "#CBD5E1"}
          linkWidth={() => 1}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={1.5}
          onNodeClick={(n: UINode) => onNodeClick(n as unknown as GraphNode)}
          // ✅ Permitir arrastrar nodos
          enableNodeDrag={true}
          // ❗ Mantener los nodos dentro del área en cada tick (por si la simulación los empuja)
          onEngineTick={() => {
            (data.nodes as unknown as UINode[]).forEach((n) => clampNodeToBounds(n));
          }}
          // ✅ Mientras se arrastra, fijar la posición y limitar a bordes
          onNodeDrag={(n: UINode) => {
            n.fx = n.x; n.fy = n.y; // fijar mientras se arrastra
            clampNodeToBounds(n);
          }}
          onNodeDragEnd={(n: UINode) => {
            // Dejar el nodo fijado donde lo soltaron (opcional: comenta para liberar)
            n.fx = n.x; n.fy = n.y;
            clampNodeToBounds(n);
          }}
          // ✅ Habilitar pan/zoom
          enableZoomInteraction={true}
          enablePanInteraction={true}
          cooldownTicks={80}
        />
      </CardContent>
    </Card>
  );
}

function ModalArticulos({ open, onOpenChange, nodo }: { open: boolean; onOpenChange: (v: boolean) => void; nodo?: GraphNode | null }) {
  if (!nodo) return null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-2xl font-bold">
            <span className="rounded-full bg-blue-600 px-3 py-1 text-sm font-medium text-white shadow-md">Término</span>
            {nodo.label}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4 pt-2">
          <h3 className="text-lg font-semibold border-b pb-1 text-slate-700">Artículos de Origen:</h3>
          {nodo.articulos?.length ? (
            <ul className="list-disc space-y-3 pl-5">
              {nodo.articulos.map((a, i) => (
                <li key={i} className="text-slate-800">
                  <a 
                    className="underline text-blue-600 hover:text-blue-800 transition duration-150 font-medium" 
                    href={a.link} 
                    target="_blank" 
                    rel="noreferrer"
                  >
                    {a.titulo}
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-slate-600 italic">No hay enlaces de documentos fuente para este nodo.</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function App() {
  const [rawData, setRawData] = useState<DataJSON | null>(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [open, setOpen] = useState(false);

  // --- LÓGICA DE BÚSQUEDA Y LLAMADA A LA API ---

  const handleSearch = async (term: string) => {
    const trimmedTerm = term.trim();
    if (!trimmedTerm) return;
    
    setLoading(true);
    setRawData(null); 
    
    const j = await fetchRAGReport(trimmedTerm);

    if (j) {
      setRawData(j);
      setQuery(trimmedTerm);
    } else {
      setRawData(defaultData);
      setQuery(trimmedTerm); // Mantener el query para el filtro aunque falle
    }
    setLoading(false);
  };
  
  // 1) Carga inicial al montar el componente (ejecuta una búsqueda predeterminada)
  useEffect(() => {
    let cancelled = false;
    const loadInitialData = async () => {
      setLoading(true);
      // Consulta de ejemplo para cargar datos al inicio
      const initialQuery = "procedimientos experimentales"; 
      
      try {
        const j = await fetchRAGReport(initialQuery);
        if (!cancelled && j) {
          setRawData(j);
          setQuery(initialQuery);
        } else if (!cancelled) {
          setRawData(defaultData); 
          setQuery(initialQuery);
        }
      } catch (e) {
        console.error("Error en carga inicial:", e);
        if (!cancelled) setRawData(defaultData);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    
    loadInitialData();
    return () => {
      cancelled = true;
    };
  }, []); 

  // 2) Construir grafo actual (filtrado por la última query)
  const graph: GraphData = useMemo(() => {
    const base = rawData ?? defaultData;
    const filtered = filterGraphByQuery(base, query); 
    return buildGraphData(filtered.grafo);
  }, [rawData, query]);

  const reporte = rawData?.reporte ?? defaultData.reporte;

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      <Navbar onSearch={handleSearch} />

      <main className="mx-auto grid grid-cols-1 gap-6 p-4 md:grid-cols-5 max-w-7xl">
        {/* Columna izquierda: Reporte */}
        <motion.section
          key={`reporte-${query}`}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="md:col-span-2"
        >
          {loading ? (
            <Card className="shadow-xl h-[400px] flex items-center justify-center">
              <CardContent className="flex flex-col items-center gap-3 p-6 text-slate-600">
                <Loader2 className="animate-spin text-blue-500" size={32} /> 
                <p className="text-lg font-medium">Generando reporte y grafo...</p>
                <p className="text-sm text-center">Consultando API en `http://127.0.0.1:8000/api/query_json`</p>
              </CardContent>
            </Card>
          ) : (
            <Reporte resumen={reporte.resumen} hallazgos={reporte.hallazgos} />)
          }
        </motion.section>

        {/* Columna derecha: Grafo */}
        <motion.section
          key={`grafo-${query}`}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="md:col-span-3"
        >
          {loading ? (
            <Card className="shadow-xl h-[624px] flex items-center justify-center">
              <CardContent className="flex items-center gap-2 p-6 text-slate-600">
                {/* Loader ya en el reporte */}
              </CardContent>
            </Card>
          ) : (
            <Grafo
              data={graph}
              onNodeClick={(n) => {
                setSelected(n);
                setOpen(true);
              }}
            />
          )}
        </motion.section>
      </main>

      {/* Modal con artículos del nodo */}
      <ModalArticulos open={open} onOpenChange={setOpen} nodo={selected} />
    </div>
  );
}
