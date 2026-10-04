export const STATUS = {
  working: { label: 'Computing', detail: 'Executing assigned work.', color: '#eea10a' },
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

export function formatData(amount) {
  return amount >= 1024 ? `${(amount / 1024).toFixed(1)} GB` : `${amount.toFixed(1)} MB`
}
