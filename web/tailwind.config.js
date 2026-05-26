/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#7b39fc',
        darkPurple: '#2b2344',
      },
      fontFamily: {
        manrope: ['Manrope', 'sans-serif'],
        cabin: ['Cabin', 'sans-serif'],
        serifDisplay: ['Instrument Serif', 'serif'],
        inter: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        purpleCta: '0 12px 32px rgba(123, 57, 252, 0.28)',
      },
    },
  },
  plugins: [],
};
