import { useState } from 'react'
import TileApp from './TileApp'
import KernelApp from './KernelApp'

export default function App() {
  const [page, setPage] = useState('pool')
  return <>
    <nav className="app-nav" aria-label="Compute mode">
      <button className={page === 'pool' ? '' : 'secondary'} onClick={() => setPage('pool')}>GPU pool · Mandelbrot</button>
      <button className={page === 'python' ? '' : 'secondary'} onClick={() => setPage('python')}>Python kernel prototype</button>
    </nav>
    {page === 'pool' ? <TileApp /> : <KernelApp />}
  </>
}
