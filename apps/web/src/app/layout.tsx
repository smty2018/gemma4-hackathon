import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./styles.css";

export const metadata: Metadata = {
  title: "Multilingual Public-Service Assistant",
  description: "A multilingual AI citizen agent powered by Gemma 4.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
