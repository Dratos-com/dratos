import type { AppProps } from 'next/app'
import { ToastProvider } from '@/components/ui/use-toast'

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ToastProvider>
      <Component {...pageProps} />
    </ToastProvider>
  )
}

export default MyApp