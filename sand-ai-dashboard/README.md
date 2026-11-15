# Sand AI — Image analyser

A responsive, single-page sand/beige image-analysis dashboard. Users upload a supported image, ask a question, and receive an AI-generated answer that is checked server-side and client-side for exactly 100 words.

## Deploy to Vercel

1. Unzip this project and open a terminal in the `sand-ai-dashboard` folder.
2. Install the Vercel CLI and sign in:

   ```bash
   npm install -g vercel
   vercel login
   ```

3. Link the folder to a new or existing Vercel project:

   ```bash
   vercel link
   ```

4. In **Vercel → Project → Settings → Environment Variables**, add `OPENAI_API_KEY`. Set its value to your OpenAI API key, choose **Secret** as the variable type, and enable it for Production (and Preview/Development too, if you will use those environments). Never paste the key into `index.html` or commit it to Git.
5. Deploy to production:

   ```bash
   vercel deploy --prod
   ```

   Vercel prints the live deployment URL when the deployment completes.

You can also add the secret from the terminal, one environment at a time. The CLI prompts for the value:

```bash
vercel env add OPENAI_API_KEY production --sensitive
vercel env add OPENAI_API_KEY preview --sensitive
vercel env add OPENAI_API_KEY development --sensitive
```

After changing environment variables, create a new deployment for the changes to take effect.

## Configuration

- `OPENAI_API_KEY` — required private OpenAI API key.
- `OPENAI_MODEL` — optional model override; defaults to `gpt-5.4-mini`.

## Notes

- Supported image formats: PNG, JPEG, WEBP and GIF.
- Upload limit: 3 MB.
- Image analysis uses the OpenAI Responses API from the Vercel serverless function at `/api/analyze`; the key stays server-side.
- Image input is sent to the model as a base64 data URL. The API endpoint asks for exactly 100 words and retries a rewrite when the response count is not 100.
- OpenAI API usage may incur charges on the account associated with the API key.
