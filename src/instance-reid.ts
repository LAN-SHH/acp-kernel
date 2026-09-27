import { baseIdOf } from "./prune.js";
import { coveredMessageIds } from "./state.js";
import type { CompressionState, CoreMessage } from "./types.js";

// deriveMessageId mints "h_" + 16 lowercase hex, optionally with a within-pass
// cluster suffix "_<n>" and/or a sub-id projection "#<tail>". These are the only
// ids that collide across turns; reserved ids (acp_summary_*, retrieved_*, host
// ids) never do and are left untouched.
const CONTENT_HASH_ROOT = /^h_([0-9a-f]{16})(?:_\d+)?(?:#.*)?$/;

function clusterRoot(id: string): string | null {
  const m = CONTENT_HASH_ROOT.exec(id);
  return m ? `h_${m[1]}` : null;
}

/**
 * Re-mint live messages whose id collides with an already-FOLDED copy of the same
 * content (billion-context #1476). Runs as the FIRST pipeline node, before
 * assign-refs and prune, so the re-minted id is what gets ref'd and what prune sees.
 *
 * Root cause: deriveMessageId's cluster counter restarts every conversion pass, so
 * text re-sent after its earlier copy was folded re-derives the SAME bare `h_…` id.
 * From that single collision: assignRefs is first-wins (`if (map.byRaw[id]) continue`)
 * and the folded original already owns `byRaw[h_…]`, so the fresh instance gets NO
 * ref; and prune drops any covered id that isn't the pinned first user message, so
 * the fresh user turn silently vanishes upstream.
 *
 * Why "covered base" is the discriminator: a base is covered iff an active block
 * lists it in effectiveMessageIds, i.e. its original is off the wire (replaced by a
 * summary). So any LIVE message carrying a covered id is a new instance, not the
 * original — and must get a distinct instance id. A NON-covered base keeps the
 * converter's numbering untouched: that is the shape where the old copy is still on
 * the wire and un-folded, where the in-pass occurrence count already keeps the pair
 * distinct, and touching the ids would only churn the prefix cache. Do not broaden
 * this to all bases.
 *
 * For each conflicting root, live instances are renumbered in arrival order to
 * root_1, root_2, … skipping any number a folded copy already claims. Deterministic
 * for a fixed (state, body): a live message whose root stays covered keeps the same
 * _k every turn (prefix-cache stable) and shifts only when a newer identical
 * instance joins the pass.
 */
export function remintCoveredLiveIds(
  messages: CoreMessage[],
  state: CompressionState,
): CoreMessage[] {
  const covered = new Set<string>();
  for (const id of coveredMessageIds(state)) covered.add(baseIdOf(id));
  if (covered.size === 0) return messages;

  const groups = new Map<string, number[]>();
  for (let i = 0; i < messages.length; i++) {
    const root = clusterRoot(messages[i]!.id);
    if (root === null) continue;
    const idxs = groups.get(root);
    if (idxs) idxs.push(i);
    else groups.set(root, [i]);
  }

  const next = [...messages];
  let changed = false;
  for (const [root, idxs] of groups) {
    // Only renumber when some live instance claims an id a folded copy owns;
    // otherwise the converter's numbering is already collision-free here.
    const conflict = idxs.some((i) => covered.has(baseIdOf(messages[i]!.id)));
    if (!conflict) continue;
    let k = 1;
    for (const i of idxs) {
      while (covered.has(`${root}_${k}`)) k++;
      const id = messages[i]!.id;
      const hash = id.indexOf("#");
      const tail = hash > 0 ? id.slice(hash) : "";
      next[i] = { ...messages[i]!, id: `${root}_${k}${tail}` };
      k++;
      changed = true;
    }
  }
  return changed ? next : messages;
}
