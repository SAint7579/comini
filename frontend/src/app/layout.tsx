import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Comini - Product Search',
  description: 'Industrial Tools & Fasteners Search',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

