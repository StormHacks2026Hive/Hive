import { useEffect, useMemo, useRef, useState } from 'react'
import qrcode from 'qrcode-generator'
import { inviteLink } from '../invite.js'
import { Icon } from './HiveScene.jsx'

export default function NetworkInvite({ network, onClose }) {
  const dialog = useRef(null)
  const [copied, setCopied] = useState(false)
  const url = inviteLink(network.id)
  const code = useMemo(() => {
    const qr = qrcode(0, 'M')
    qr.addData(url)
    qr.make()
    const count = qr.getModuleCount()
    let path = ''
    for (let row = 0; row < count; row++) {
      for (let col = 0; col < count; col++) {
        if (qr.isDark(row, col)) path += `M${col + 4} ${row + 4}h1v1h-1z`
      }
    }
    return { path, size: count + 8 }
  }, [url])
  useEffect(() => { dialog.current.showModal() }, [])

  async function copyLink() {
    try { await navigator.clipboard.writeText(url); setCopied(true) }
    catch { setCopied(false) }
  }

  return (
    <dialog ref={dialog} className="network-invite confirm-dialog" aria-labelledby="invite-title"
      onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose() }}>
      <button className="icon-button invite-close" aria-label="Close invite" onClick={onClose} autoFocus><Icon name="close" /></button>
      <h2 id="invite-title">Join {network.name}</h2>
      <svg className="invite-qr" viewBox={`0 0 ${code.size} ${code.size}`} role="img" aria-label="Scan to join this network as a guest" shapeRendering="crispEdges">
        <rect width={code.size} height={code.size} fill="#fff" />
        <path d={code.path} fill="#000" />
      </svg>
      <p>Scan, enter the password, join.</p>
      <a className="invite-url" href={url}>hivehacks.tech</a>
      <span className="mono invite-network-id">{network.id}</span>
      <button className="button button-secondary" onClick={copyLink}><Icon name="copy" />{copied ? 'Copied' : 'Copy invite link'}</button>
      <span className="sr-only" role="status">{copied ? 'Invite link copied' : ''}</span>
    </dialog>
  )
}
