# Hive frontend

A React workspace with Google sign-in inside a honeycomb that expands into the
app, local network controls, and an interactive honeycomb node map. The backend
does not need to be running.

## Run locally

From the repository root:

```sh
cd HiveFrontend
npm ci
npm run dev
```

Open **http://localhost:5173**.

Vite reads the public `GOOGLE_CLIENT_ID` from the repository root `.env`.
You can instead override it with `VITE_GOOGLE_CLIENT_ID` in
`HiveFrontend/.env.local` or your shell. Only the public client ID is included
in the frontend; no Google client secret is needed. Restart Vite after changing
environment variables. A real client ID is required to sign in.

## Configure Google

Follow [Google's setup guide](https://developers.google.com/identity/gsi/web/guides/get-google-api-clientid)
to create an OAuth client with application type **Web application**. Add these
**Authorized JavaScript origins**:

- `http://localhost`
- `http://localhost:5173`

If you open the app using `127.0.0.1`, also authorize the matching origin.
Configure the consent screen and add your Google account as a test user if the
Google app is in testing mode. This uses the popup flow; no redirect URI is
needed. Put the client ID in the root `.env`:

```dotenv
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
```

Alternatively, use `HiveFrontend/.env.local`:

```dotenv
VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
```

## Frontend behavior

`GoogleSignIn.jsx` renders Google's official button. The callback decodes the
returned ID token to display the account name and email. Profile claims are
checked for expected audience, issuer, and expiration, but the token signature
is not verified. This is only a frontend account preview; it does not establish
an authenticated backend session or authorize access to any API.

Profile data stays in React memory, so refreshing the page resets sign-in.
Tokens are not persisted. Sign out clears the profile and disables Google's
automatic account selection; it leaves your Google account signed in.
There are no requests to `/auth`.

After sign-in, the Network tab lets you join with a network ID and password or
create a network with a name and password (at least eight characters). Creating
a network gives you an ID to copy and share. Networks created during this visit
can be rejoined with their matching password; other IDs open a local preview.
This does not create or connect to a real network.

The workspace includes pause/resume, turn off/start, leave-network controls,
a compute contribution slider, and an incoming-data switch. The Mapping tab
shows seven sample nodes with simulated receiving, sending, idle, paused, and
offline states. Select a comb to inspect its transfer rate and received data,
or use the filters and zoom controls. Only your own node has controls. Activity
updates every three seconds. Node transfer stops immediately when paused or
switched off. The animation respects reduced-motion preferences.

Networks, passwords, profile data, and activity stay in memory. Refreshing or
signing out resets the workspace. No credentials or passwords are persisted.

When backend authentication is added later, verify the ID token on the server
and establish a server session before granting access to protected resources.

## Checks

```sh
npm run lint
npm test
npm run build
```

`npm run preview` also works without a backend. Authorize its origin in Google
Cloud if you use it to test sign-in. Production origins must be authorized too.
