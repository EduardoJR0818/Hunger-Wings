import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
// Eliminamos la dependencia externa ForceGraph2D y sus tipos asociados para asegurar la compilación.
// import ForceGraph2D from "react-force-graph-2d";
// import type { ForceGraphMethods, NodeObject } from "react-force-graph-2d";
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

// ⚠️ Componente Grafo simplificado (SVG estático) para asegurar la compilación
function Grafo({ data, onNodeClick }: { data: GraphData; onNodeClick: (n: GraphNode) => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { nodes, links } = data;

  // Mapa para posicionamiento simple (estático)
  const nodePositions = useMemo(() => {
    const positions = new Map<string, { x: number, y: number }>();
    const angleStep = (2 * Math.PI) / nodes.length;
    const radius = 250;
    nodes.forEach((node, i) => {
      positions.set(node.id, {
        x: 300 + radius * Math.cos(i * angleStep),
        y: 250 + radius * Math.sin(i * angleStep),
      });
    });
    return positions;
  }, [nodes]);
  
  return (
    <Card className="shadow-xl border border-slate-200">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-3">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-network text-blue-600"><rect x="16" y="2" width="6" height="6" rx="1"/><path d="M4 17v-7a3 3 0 0 1 3-3h2"/><path d="M15 22l-5-5-5 5"/><path d="m10 17 5 5"/></svg>
          <CardTitle className="text-2xl font-bold text-slate-800">Grafo de conocimiento (Vista simplificada)</CardTitle>
        </div>
        <div className="text-sm text-slate-500">Haz clic en un nodo para ver artículos.</div>
      </CardHeader>
      <CardContent className="h-[560px] p-0" ref={containerRef}>
        <svg width="100%" height="100%" viewBox="0 0 600 500">
          {/* Líneas (Links) */}
          {links.map((link, i) => {
            const sourcePos = nodePositions.get(link.source);
            const targetPos = nodePositions.get(link.target);
            if (!sourcePos || !targetPos) return null;
            return (
              <line
                key={i}
                x1={sourcePos.x}
                y1={sourcePos.y}
                x2={targetPos.x}
                y2={targetPos.y}
                stroke="#94A3B8"
                strokeWidth="1.5"
              />
            );
          })}
          
          {/* Nodos (Nodes) */}
          {nodes.map((node) => {
            const pos = nodePositions.get(node.id);
            if (!pos) return null;
            
            const fontSize = 12;
            const textLength = node.label.length;
            const width = Math.max(70, textLength * 7 + 10); // Ancho basado en texto
            const height = 24;
            
            return (
              <g 
                key={node.id} 
                transform={`translate(${pos.x}, ${pos.y})`}
                onClick={() => onNodeClick(node)}
                className="cursor-pointer"
              >
                {/* Fondo del nodo (Pastilla redondeada) */}
                <rect
                  x={-width / 2}
                  y={-height / 2}
                  rx="6"
                  ry="6"
                  width={width}
                  height={height}
                  fill="#0f172a"
                  className="hover:fill-blue-600 transition duration-150"
                />
                {/* Texto del nodo */}
                <text
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fontSize: `${fontSize}px`, fill: 'white', fontFamily: 'Inter, system-ui' }}
                  y={1} // Ajuste vertical
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
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
