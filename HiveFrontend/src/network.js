export const STATUS = {
  receiving: {
    label: 'Receiving',
    detail: 'Actively receiving data from the network.',
    color: '#eea10a',
  },
  sending: { label: 'Sending', detail: 'Sharing data with another node.', color: '#cf5900' },
  idle: {
    label: 'Idle',
    detail: 'Online and ready. No data is moving right now.',
    color: '#ffd16a',
  },
  paused: {
    label: 'Paused',
    detail: 'Connected to the network. Data transfer is paused.',
    color: '#bd7718',
  },
  offline: {
    label: 'Offline',
    detail: 'This node is switched off and is not transferring data.',
    color: '#b5aa91',
  },
}

export function createPreviewNodes(name) {
  return [
    {
      id: 'you',
      name: `${name.split(' ')[0]}'s node`,
      short: 'YOU',
      x: 410,
      y: 290,
      status: 'receiving',
      rate: 2.4,
      own: true,
      device: 'This browser',
      received: 0,
    },
    {
      id: 'amber',
      name: 'Amber',
      short: 'AM',
      x: 235,
      y: 180,
      status: 'sending',
      rate: 1.8,
      device: 'Desktop',
      received: 12.8,
    },
    {
      id: 'clover',
      name: 'Clover',
      short: 'CL',
      x: 570,
      y: 175,
      status: 'receiving',
      rate: 3.2,
      device: 'Laptop',
      received: 24.1,
    },
    {
      id: 'fern',
      name: 'Fern',
      short: 'FE',
      x: 610,
      y: 360,
      status: 'idle',
      rate: 0,
      device: 'Desktop',
      received: 8.4,
    },
    {
      id: 'pollen',
      name: 'Pollen',
      short: 'PO',
      x: 430,
      y: 460,
      status: 'offline',
      rate: 0,
      device: 'Laptop',
      received: 5.6,
    },
    {
      id: 'sage',
      name: 'Sage',
      short: 'SA',
      x: 215,
      y: 410,
      status: 'receiving',
      rate: 1.6,
      device: 'Desktop',
      received: 17.3,
    },
    {
      id: 'willow',
      name: 'Willow',
      short: 'WI',
      x: 110,
      y: 275,
      status: 'idle',
      rate: 0,
      device: 'Laptop',
      received: 9.2,
    },
  ]
}

export function advancePreview(nodes, tick, { mode, accepting, capacity }) {
  const phases = ['receiving', 'idle', 'sending', 'receiving', 'sending', 'idle']
  return nodes.map((node, index) => {
    if (node.status === 'offline' && !node.own) return node
    let status =
      node.own && mode !== 'running'
        ? mode === 'off'
          ? 'offline'
          : 'paused'
        : phases[(tick + index * 2) % phases.length]
    if (node.own && !accepting && status === 'receiving') status = 'idle'
    const rate = ['receiving', 'sending'].includes(status)
      ? Number(((1.2 + ((tick + index) % 5) * 0.6) * (node.own ? capacity / 100 : 1)).toFixed(1))
      : 0
    return {
      ...node,
      status,
      rate,
      received: node.received + (status === 'receiving' ? rate * 3 : 0),
    }
  })
}

export function formatData(amount) {
  return amount >= 1024 ? `${(amount / 1024).toFixed(1)} GB` : `${amount.toFixed(1)} MB`
}
