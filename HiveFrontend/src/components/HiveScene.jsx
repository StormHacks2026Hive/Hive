export function HexIcon({ className = '', ...props }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" aria-hidden="true" {...props}>
      <path d="m16 2 12 7v14l-12 7L4 23V9Z" stroke="currentColor" strokeWidth="1.8" />
      <path d="m16 9 6 3.5v7L16 23l-6-3.5v-7Z" fill="currentColor" />
    </svg>
  )
}

export function Icon({ name, ...props }) {
  const paths = {
    network: (
      <>
        <circle cx="12" cy="5" r="2" />
        <circle cx="5" cy="18" r="2" />
        <circle cx="19" cy="18" r="2" />
        <path d="M12 7v5m-7 4v-4h14v4" />
      </>
    ),
    map: (
      <>
        <path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2Z" />
        <path d="M9 3v16m6-14v16" />
      </>
    ),
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    plus: <path d="M12 5v14M5 12h14" />,
    power: (
      <>
        <path d="M12 3v8m-5-6a9 9 0 1 0 10 0" />
      </>
    ),
    pause: (
      <>
        <path d="M8 5v14m8-14v14" />
      </>
    ),
    play: <path d="m8 5 11 7-11 7Z" />,
    copy: (
      <>
        <rect x="8" y="8" width="12" height="12" rx="2" />
        <path d="M16 8V4H4v12h4" />
      </>
    ),
    eye: (
      <>
        <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    lock: (
      <>
        <rect x="5" y="10" width="14" height="11" rx="2" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3" />
      </>
    ),
    close: <path d="m6 6 12 12M6 18 18 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    logout: (
      <>
        <path d="M9 4H4v16h5m0-8h12m-4-4 4 4-4 4" />
      </>
    ),
    down: <path d="M12 4v16m-6-6 6 6 6-6" />,
    up: <path d="M12 20V4m-6 6 6-6 6 6" />,
  }
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name] || paths.network}
    </svg>
  )
}

// The tree belongs to the illustrated login scene.
export function TreeArt() {
  return (
    <>
      <defs>
        <pattern id="bark-speckles" width="38" height="49" patternUnits="userSpaceOnUse">
          <path
            d="m4 5 1 3m13 3-1 4m14-9-1 3M8 24l1 5m14-8-1 7M31 37l-2 3M14 42l1 4"
            stroke="#918573"
            strokeWidth="1.4"
            opacity=".65"
          />
          <path d="m6 13 1 2m19 22 1 2M35 19l-1 3M18 30l-1 2" stroke="#b4a38b" strokeWidth="2" />
        </pattern>
      </defs>
      <path
        d="M788-12c6 47 10 85-8 109-33 44-75 7-131-9-100-28-185-26-305-11C204 95 184 71 164 38c-11-18-17-7-7 10 35 52 20 53-14 39C88 64 54 52 21 52-9 54 27 63 47 69c80 28 102 43 163 45 76 1 166 18 270 2 84-12 119 4 170 30 59 31 96 41 114 107 11 43 14 166 11 244h81V-12Z"
        fill="#d7cbb5"
      />
      <path
        d="M803 0c-5 31-15 85-28 114-12 29-52 24-77 13-69-31-121-43-196-37-108 8-166 9-244 9-45 0-70-6-83-20 23 37 72 43 158 40 79-3 138-20 206-9 105 18 126 72 210 61 44 55 31 210 35 322h34c-8-115-15-223-11-318 2-55 12-124 23-175Z"
        fill="#b9ad99"
        opacity=".78"
      />
      <path
        d="M823-5c-7 70-21 122-27 190-4 57 6 143 7 213m-6-173c-9 49-8 104-5 156M832 276c-4 56 0 130-7 195M810 24l-4 30M60 72c62 18 87 33 155 33m195 8c79-11 145-22 226 16"
        fill="none"
        stroke="#9f927f"
        strokeWidth="5"
        strokeLinecap="round"
        opacity=".54"
      />
      <path
        d="M0-4h63c20 31 24 70-4 101-12 14-35 21-59 10Z"
        fill="#b7b198"
        stroke="#eee2c9"
        strokeWidth="2"
      />
      <path
        d="M0-4h63c20 31 24 70-4 101-12 14-35 21-59 10ZM788-12c6 47 10 85-8 109-33 44-75 7-131-9-100-28-185-26-305-11C204 95 184 71 164 38c-11-18-17-7-7 10 35 52 20 53-14 39C88 64 54 52 21 52-9 54 27 63 47 69c80 28 102 43 163 45 76 1 166 18 270 2 84-12 119 4 170 30 59 31 96 41 114 107 11 43 14 166 11 244h81V-12Z"
        fill="url(#bark-speckles)"
      />
    </>
  )
}

export function HiveBody({ doorRadius = 113, shaded = true }) {
  return (
    <g stroke="#5c421b" strokeWidth="2.5" strokeLinejoin="round">
      <ellipse cx="200" cy="346" rx="121" ry="76" fill="#945911" />
      <path
        d="M55 288c0-43 63-72 145-72s145 29 145 72v35c0 41-59 64-145 64S55 364 55 323Z"
        fill="#ba7618"
      />
      <path
        d="M320 272c23 28 18 81-23 98-34 14-83 19-122 14 102 20 170-7 170-61v-35Z"
        fill="#945911"
        stroke="none"
      />
      <ellipse cx="200" cy="109" rx="121" ry="80" fill="#945911" />
      <path d="M50 173c0-44 66-79 150-79s150 35 150 79v33H50Z" fill="#ba7618" />
      <path d="M50 173c0-44 66-79 150-79s150 35 150 79" fill="none" />
      <path
        d="M15 231c0-57 62-96 185-96s185 39 185 96c0 73-37 91-185 91S15 304 15 231Z"
        fill="#bd7718"
      />
      <path
        d="M331 157c40 46 41 123 5 147-22 14-52 15-68 16 89-4 117-24 117-89 0-33-20-58-54-74Z"
        fill="#9b6012"
        stroke="none"
      />
      <path
        d="M15 231c0-57 62-96 185-96s185 39 185 96c0 73-37 91-185 91S15 304 15 231Z"
        fill="none"
      />
      <circle className="hive-door" cx="200" cy="232" r={doorRadius} fill="#ffd16a" strokeWidth="2" />
      {shaded && <path
        className="hive-door-shade"
        d="M211 120c105 15 133 158 26 221 121-33 119-208-26-221Z"
        fill="#eea10a"
        stroke="none"
      />}
    </g>
  )
}

export function HangingHive() {
  return (
    <svg className="hive-illustration" viewBox="0 0 400 450" aria-hidden="true">
      <path d="m206 2-5 69" stroke="#493920" strokeWidth="12" strokeLinecap="round" />
      <HiveBody />
    </svg>
  )
}

export default function HiveScene() {
  return (
    <svg className="hive-scene" viewBox="0 0 846 486" preserveAspectRatio="none" aria-hidden="true">
      <TreeArt />
    </svg>
  )
}
