import React from 'react';
import { createRoot } from 'react-dom/client';
import { ChevronDown, Menu, X } from 'lucide-react';
import './styles.css';

const videoUrl =
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260210_031346_d87182fb-b0af-4273-84d1-c6fd17d6bf0f.mp4';

const navItems = [
  { label: 'Home' },
  { label: 'Services', hasMenu: true },
  { label: 'Reviews' },
  { label: 'Contact us' },
];

function Logo() {
  return (
    <a aria-label="Future home" className="flex items-center" href="#">
      <svg
        aria-hidden="true"
        className="h-8 w-8"
        fill="none"
        viewBox="0 0 32 32"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M1.04356 6.35771L13.6437 0.666504L30.9564 8.38109L18.3563 14.0723L1.04356 6.35771Z"
          fill="white"
        />
        <path
          d="M1.04356 15.6908L13.6437 9.99963L30.9564 17.7142L18.3563 23.4054L1.04356 15.6908Z"
          fill="white"
          opacity="0.76"
        />
        <path
          d="M1.04356 25.0239L13.6437 19.3328L30.9564 27.0473L18.3563 32.7385L1.04356 25.0239Z"
          fill="white"
          opacity="0.52"
        />
      </svg>
    </a>
  );
}

function Navbar({ menuOpen, setMenuOpen }) {
  return (
    <header className="relative z-20 w-full bg-transparent px-6 py-4 md:px-[120px]">
      <nav className="mx-auto flex h-11 max-w-[1440px] items-center justify-between">
        <div className="flex items-center gap-10">
          <Logo />

          <div className="hidden items-center gap-8 lg:flex">
            {navItems.map((item) => (
              <a
                className="flex items-center gap-1 font-manrope text-sm font-medium text-white transition hover:opacity-80"
                href="#"
                key={item.label}
              >
                {item.label}
                {item.hasMenu ? <ChevronDown aria-hidden="true" size={16} strokeWidth={2} /> : null}
              </a>
            ))}
          </div>
        </div>

        <div className="hidden items-center gap-3 lg:flex">
          <a
            className="rounded-lg border border-[#d4d4d4] bg-white px-5 py-2.5 font-manrope text-sm font-semibold text-[#171717] transition hover:bg-white/90"
            href="#"
          >
            Sign In
          </a>
          <a
            className="rounded-lg bg-primary px-5 py-2.5 font-manrope text-sm font-semibold text-[#fafafa] shadow-purpleCta transition hover:bg-[#8a50ff]"
            href="#"
          >
            Get Started
          </a>
        </div>

        <button
          aria-expanded={menuOpen}
          aria-label="Toggle navigation menu"
          className="grid h-11 w-11 place-items-center rounded-lg text-white transition hover:bg-white/10 lg:hidden"
          onClick={() => setMenuOpen((open) => !open)}
          type="button"
        >
          {menuOpen ? <X size={26} /> : <Menu size={28} />}
        </button>
      </nav>
    </header>
  );
}

function MobileMenu({ open, setOpen }) {
  return (
    <div
      className={`fixed inset-0 z-30 flex flex-col bg-black px-6 py-5 transition duration-300 lg:hidden ${
        open ? 'visible opacity-100' : 'invisible opacity-0'
      }`}
    >
      <div className="flex items-center justify-between">
        <Logo />
        <button
          aria-label="Close navigation menu"
          className="grid h-11 w-11 place-items-center rounded-lg text-white transition hover:bg-white/10"
          onClick={() => setOpen(false)}
          type="button"
        >
          <X size={26} />
        </button>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center gap-8">
        {navItems.map((item) => (
          <a
            className="flex items-center gap-2 font-manrope text-2xl font-medium text-white"
            href="#"
            key={item.label}
            onClick={() => setOpen(false)}
          >
            {item.label}
            {item.hasMenu ? <ChevronDown aria-hidden="true" size={22} /> : null}
          </a>
        ))}
      </div>

      <div className="grid gap-3 pb-4">
        <a
          className="rounded-lg border border-[#d4d4d4] bg-white px-5 py-3 text-center font-manrope text-sm font-semibold text-[#171717]"
          href="#"
          onClick={() => setOpen(false)}
        >
          Sign In
        </a>
        <a
          className="rounded-lg bg-primary px-5 py-3 text-center font-manrope text-sm font-semibold text-[#fafafa]"
          href="#"
          onClick={() => setOpen(false)}
        >
          Get Started
        </a>
      </div>
    </div>
  );
}

function Hero() {
  return (
    <main className="relative z-10 flex min-h-[calc(100vh-76px)] flex-col items-center justify-center px-6 pb-12 pt-20 text-center md:px-10">
      <div className="mt-32 flex max-w-[1040px] flex-col items-center">
        <div className="flex h-[38px] items-center gap-2 rounded-[10px] border border-[rgba(164,132,215,0.5)] bg-[rgba(85,80,110,0.4)] px-2.5 font-cabin text-sm font-medium text-white backdrop-blur-md">
          <span className="rounded-md bg-primary px-2.5 py-1 text-white">New</span>
          <span>Say Hello to Datacore v3.2</span>
        </div>

        <h1 className="mt-7 max-w-[1010px] font-serifDisplay text-5xl leading-[1.1] text-white sm:text-6xl md:text-7xl lg:text-[96px]">
          Book your perfect stay instantly{' '}
          <em className="mx-1 italic md:mx-3">and</em>
          hassle-free
        </h1>

        <p className="mt-7 max-w-[662px] font-inter text-lg font-normal leading-8 text-white/70">
          Discover handpicked hotels, resorts, and stays across your favorite destinations.
          Enjoy exclusive deals, fast booking, and 24/7 support.
        </p>

        <div className="mt-9 flex w-full max-w-[420px] flex-col gap-3 sm:w-auto sm:max-w-none sm:flex-row">
          <a
            className="rounded-[10px] bg-primary px-7 py-4 font-cabin text-base font-medium text-white transition hover:bg-[#8a50ff]"
            href="#"
          >
            Book a Free Demo
          </a>
          <a
            className="rounded-[10px] bg-darkPurple px-7 py-4 font-cabin text-base font-medium text-[#f6f7f9] transition hover:bg-[#3b315c]"
            href="#"
          >
            Get Started Now
          </a>
        </div>
      </div>
    </main>
  );
}

function App() {
  const [menuOpen, setMenuOpen] = React.useState(false);

  React.useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [menuOpen]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-darkPurple">
      <video
        aria-hidden="true"
        autoPlay
        className="absolute inset-0 h-full min-h-screen w-full object-cover"
        loop
        muted
        playsInline
        preload="auto"
        src={videoUrl}
      />

      <Navbar menuOpen={menuOpen} setMenuOpen={setMenuOpen} />
      <Hero />
      <MobileMenu open={menuOpen} setOpen={setMenuOpen} />
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
