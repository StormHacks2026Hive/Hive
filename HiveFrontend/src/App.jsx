import { useState } from 'react'
import Dashboard from './components/Dashboard'
import TileApp from './TileApp'
import KernelApp from './KernelApp'

export default function App() {
  const [page, setPage] = useState('workloads')
  return <>
    <nav className="app-nav" aria-label="Compute mode">
      <a className="brand" href="/">hive<span>.</span></a>
      <button className={page === 'workloads' ? '' : 'secondary'} onClick={() => setPage('workloads')}>Distributed compute</button>
      <button className={page === 'pool' ? '' : 'secondary'} onClick={() => setPage('pool')}>Mandelbrot</button>
      <button className={page === 'python' ? '' : 'secondary'} onClick={() => setPage('python')}>Python prototype</button>
    </nav>
    {page === 'workloads' ? <Dashboard /> : page === 'pool' ? <TileApp /> : <KernelApp />}
  </>
}
