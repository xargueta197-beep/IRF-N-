import type { Metadata } from "next";
import { DM_Sans, Instrument_Serif } from "next/font/google";
import Link from "next/link";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import { colors } from "@/lib/theme";
import "./globals.css";

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-instrument-serif",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
});

export const metadata: Metadata = {
  title: "IRF-N — Índice de Régimen Filtrado con Noticias",
  description: "Indicador de investigación de regímenes de mercado. Araht Analytics.",
};

const NAV_LINKS = [
  { href: "/", label: "Hoy" },
  { href: "/historico", label: "Histórico" },
  { href: "/metodologia", label: "Metodología" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es" className={`${instrumentSerif.variable} ${dmSans.variable}`}>
      <body className="font-body flex min-h-screen flex-col" style={{ backgroundColor: colors.background, color: colors.foreground }}>
        <header className="flex items-center justify-between border-b px-6 py-4" style={{ borderColor: colors.border }}>
          <Link href="/" className="font-display text-2xl">
            IRF-N
          </Link>
          <nav className="flex gap-6">
            {NAV_LINKS.map((l) => (
              <Link key={l.href} href={l.href} className="font-body text-sm" style={{ color: colors.foreground }}>
                {l.label}
              </Link>
            ))}
          </nav>
        </header>
        <main className="flex-1 px-6 py-8">{children}</main>
        <DisclaimerBanner />
      </body>
    </html>
  );
}
