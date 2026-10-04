import { useCallback, useRef } from 'react'
import { flushSync } from 'react-dom'

export default function useUITransition() {
  const current = useRef(null)
  return useCallback((update, { page = false } = {}) => {
    current.current?.skipTransition()
    if (document.visibilityState === 'hidden' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      update()
      return
    }
    if (!document.startViewTransition) {
      const outgoing = page && document.querySelector('.workspace-content')
      if (!outgoing) {
        update()
        return
      }
      let committed = false
      let skipped = false
      const commit = () => {
        if (!committed) {
          committed = true
          flushSync(update)
        }
      }
      const animation = outgoing.animate([{ opacity: 1 }, { opacity: 0, transform: 'translateY(-4px)' }], { duration: 160, easing: 'ease-in', fill: 'forwards' })
      const fallback = { skipTransition() { skipped = true; animation.cancel(); commit() } }
      current.current = fallback
      animation.finished.catch(() => {}).then(() => {
        if (skipped) return
        commit()
        animation.cancel()
        const incoming = document.querySelector('.workspace-content')?.animate(
          [{ opacity: 0, transform: 'translateY(10px)' }, { opacity: 1, transform: 'translateY(0)' }],
          { duration: 420, easing: 'cubic-bezier(.22, 1, .36, 1)' },
        )
        fallback.skipTransition = () => incoming?.cancel()
        incoming?.finished.catch(() => {}).finally(() => {
          if (current.current === fallback) current.current = null
        })
      })
      return
    }
    document.documentElement.dataset.pageTransition = String(page)
    const transition = document.startViewTransition(() => flushSync(update))
    current.current = transition
    transition.ready.catch(() => {})
    transition.finished.catch(() => {}).finally(() => {
      if (current.current === transition) {
        current.current = null
        delete document.documentElement.dataset.pageTransition
      }
    })
  }, [])
}
