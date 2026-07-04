const examples = [
  "Explain this electricity bill in Bengali",
  "What is the last date to pay?",
  "Explain it like you are speaking to my grandmother",
];

export default function HomePage() {
  return (
    <main>
      <section className="hero">
        <p className="eyebrow">Multilingual public-service assistant</p>
        <h1>Understand the document. Know the next step.</h1>
        <p className="lede">
          Upload a public-service document and receive a clear, multilingual,
          evidence-grounded action plan.
        </p>
      </section>

      <section className="card" aria-labelledby="upload-title">
        <div>
          <p className="step">Step 1</p>
          <h2 id="upload-title">Add a document</h2>
          <p>Images and PDFs stay local until you explicitly submit them.</p>
        </div>
        <label className="upload">
          <span>Choose an image or PDF</span>
          <input type="file" accept="image/*,.pdf,application/pdf" />
        </label>
      </section>

      <section className="examples" aria-labelledby="examples-title">
        <p className="step">Try asking</p>
        <h2 id="examples-title">Speak naturally</h2>
        <div className="chips">
          {examples.map((example) => (
            <button key={example} type="button">
              {example}
            </button>
          ))}
        </div>
      </section>

      <aside className="safety">
        The assistant shows its evidence and asks before creating reminders,
        opening maps, or taking any external action.
      </aside>
    </main>
  );
}
