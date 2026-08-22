import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "VirtualBackground Lab",
  description: "An on-device virtual background and segmentation processing workbench.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
