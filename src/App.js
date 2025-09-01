import { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/query", { question: query });
      setResponse(res.data.answer);
    } catch (err) {
      setResponse("Error: " + err.message);
    }
    setLoading(false);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Resume QA Bot</h1>
      <textarea
        className="w-full p-2 border border-gray-300 rounded mb-2"
        rows={4}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question about the resume..."
      />
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded"
        onClick={handleAsk}
        disabled={loading}
      >
        {loading ? "Thinking..." : "Ask"}
      </button>
      <div className="mt-4 p-4 border rounded bg-gray-50">
        <strong>Answer:</strong>
        <p>{response}</p>
      </div>
    </div>
  );
}

export default App;
