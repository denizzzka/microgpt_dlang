/*
The most atomic way to train and run inference for a GPT in pure.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
https://karpathy.github.io/2026/02/12/microgpt/

D language version by Denis Feklushkin
https://github.com/denizzzka/
*/

import std;

void main()
{
    auto rng = MinstdRand0(42);

    // Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
    //TODO: original code performs http-request, implement same?
    const docs = readText("names.txt")
        .splitLines
        .randomShuffle(rng)
        .array;

    writeln("num docs: ", docs.length);

    // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    const uchars = docs.join.array.sort.uniq.array;
    const BOS = uchars.length; /// token id for a special Beginning of Sequence (BOS) token
    const vocab_size = uchars.length + 1; /// total number of unique tokens, +1 is for BOS
    writeln("vocab size: ", vocab_size);

    // Let there be Autograd to recursively apply the chain rule through a computation graph
    static class Value
    {
        float data;
        float grad;
        private Value[] _children; //TODO: remove underscore
        private float[] _local_grads;

        this(float data, Value[] children = null, float[] local_grads = null) pure
        {
            this.data = data;                // scalar value of this node calculated during forward pass
            this.grad = 0;                   // derivative of the loss w.r.t. this node, calculated in backward pass
            this._children = children;       // children of this node in the computation graph
            this._local_grads = local_grads; // local derivative of this node w.r.t. its children
        }

        auto opBinary(string s)(float other) if(s != "^^") => opBinary!s(new Value(other));

        auto opBinary(string s)(Value other) pure if(s == "+") => new Value(this.data + other.data, [this, other], [1, 1]);
        auto opBinary(string s)(Value other) if(s == "*") => new Value(this.data * other.data, [this, other], [other.data, this.data]);
        auto opBinary(string s)(float other) if(s == "^^") => new Value(this.data ^^ other, [this], [other * this.data ^^ (other-1)]);
        auto log() => new Value(std.math.log(data), [this], [1.0f / data]);
        auto exp() => new Value(std.math.exp(data), [this], [std.math.exp(data)]);
        auto relu() => new Value(data < 0 ? 0 : data, [this], [data < 0 ? 0 : 1]);
        auto opUnary(string s)() if(s == "-") => this * -1;
        //~ def __radd__(self, other): return self + other
        //~ def __sub__(self, other): return self + (-other)
        auto opBinary(string s)(Value other) if(s == "-") => this + (-other);
        //~ def __rsub__(self, other): return other + (-self)
        //~ def __rmul__(self, other): return self * other
        auto opBinary(string s)(Value other) if(s == "/") => this * other^^-1;
        //~ def __rtruediv__(self, other): return other * self**-1

        void backward()
        {
            Value[] topo;
            bool[const Value] visited;

            void buildTopo(Value v)
            {
                if(!(v in visited))
                {
                    visited[v] = true;

                    foreach(child; v._children)
                        buildTopo(child);

                    topo ~= v;
                }
            }
            buildTopo(this);

            grad = 1;
            foreach_reverse (v; topo)
                foreach (i, child; v._children)
                    child.grad += v._local_grads[i] * v.grad;
        }
    }

    // Initialize the parameters, to store the knowledge of the model
    const n_layer = 1;      /// depth of the transformer neural network (number of layers)
    const n_embd = 16;      /// width of the network (embedding dimension)
    const block_size = 16;  /// maximum context length of the attention window (note: the longest name is 15 characters)
    const n_head = 4;       /// number of attention heads
    const head_dim = n_embd / n_head;   /// derived dimension of each head

    alias Matrix = Value[][];
    Matrix matrix(size_t nout, uint nin, float std=0.08) pure
    {
        Value[][] ret;
        ret.length = nout;

        foreach(ref row; ret)
        {
            row.length = nin;

            foreach(ref cell; row)
                cell = new Value(randomGauss(rng, std));
        }

        return ret;
    }

    static auto getAllParams(Matrix matrix) => matrix.joiner;

    Matrix wte = matrix(vocab_size, n_embd);
    Matrix wpe = matrix(block_size, n_embd);
    Matrix lm_head = matrix(vocab_size, n_embd);

    /// flatten params into a single list
    Value[] params = [
        getAllParams(wte), getAllParams(wpe), getAllParams(lm_head),
    ].join;

    class Layer
    {
        union
        {
            struct
            {
                Matrix attn_wq;
                Matrix attn_wk;
                Matrix attn_wv;
                Matrix attn_wo;

                Matrix mlp_fc1;
                Matrix mlp_fc2;
            }

            private Matrix[6] allMat;
        }

        Value[] getAll() => allMat[].joiner.join;

        this() pure
        {
            attn_wq = matrix(n_embd, n_embd);
            attn_wk = matrix(n_embd, n_embd);
            attn_wv = matrix(n_embd, n_embd);
            attn_wo = matrix(n_embd, n_embd);

            mlp_fc1 = matrix(4 * n_embd, n_embd);
            mlp_fc2 = matrix(n_embd, 4 * n_embd);
        }
    }

    Layer[n_layer] layers;
    foreach(ref l; layers)
    {
        l = new Layer;
        params ~= l.getAll;
    }

    writeln("num params: ", params.length);

    // Define the model architecture: a function mapping tokens and parameters to logits over what comes next
    // Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
    static Value[] linear(Value[] x, Matrix weights)
    {
        return weights.map!(
            (w) => zip(x, w)
                .map!((e) => e[0] * e[1])
                .sumVals
        ).array;
    }

    auto softmax(Value[] logits)
    {
        Value max_val = logits.maxElement!((a) => a.data);
        auto exps = logits.map!((val) => (val - max_val).exp).array;
        Value total = exps.sumVals;
        return exps.map!((e) => e / total);
    }

    //TODO: maybe it is worth to return array
    static auto rmsnorm(R)(R x) pure
    {
        auto ms = x.map!"a*a".sumVals / x.length;
        auto scale = (ms + 1e-5) ^^ -0.5; //TODO: use float.min_normal instead of 1e-5?
        return x.map!((a) => a * scale);
    }

    Value[] gpt(in ushort token_id, in ushort pos_id, Matrix keys, Matrix values)
    {
        auto tok_emb = wte[token_id];
        auto pos_emb = wpe[token_id];

        auto x_RENAMEME = zip(tok_emb, pos_emb).map!((e) => e[0] + e[1]); // joint token and position embedding
        auto x = rmsnorm(x_RENAMEME).array; // note: not redundant due to backward pass via the residual connection

        foreach(ref li; layers)
        {
            // 1) Multi-head Attention block
            auto x_residual = x;
            x = rmsnorm(x).array;

            auto q = linear(x, li.attn_wq);
            auto k = linear(x, li.attn_wk);
            auto v = linear(x, li.attn_wv);

            keys ~= k;
            values ~= v;

            Value[] x_attn;
            foreach(h; 0 .. n_head)
            {
                // Slice out this head's portion of q, k, v
                const hs = h * head_dim;
                //TODO: remove magic:
                auto q_h = q[hs .. hs+head_dim];
                auto k_h = keys.map!((e) => e[hs .. hs + head_dim]);
                auto v_h = values.map!((e) => e[hs .. hs + head_dim]);

                // Dot product of query against all past keys, scaled to prevent vanishing gradients
                Value[] attn_logits;
                const divider = head_dim ^^ 0.5f;
                foreach(t; 0 .. k_h.length)
                {
                    auto s = head_dim.iota.map!((j) => q_h[j] * k_h[t][j]).sumVals;
                    attn_logits ~= s / divider;
                }

                auto attn_weights = softmax(attn_logits);
                auto head_out = v_h.map!((vh_j) =>
                    zip(attn_weights, vh_j)
                        .map!((e) => e[0] * e[1])
                        .sumVals
                );

                x_attn ~= head_out.array;
            }

            // TODO: Why x used again?
            x = linear(x_attn, li.attn_wo);
            x = zip(x, x_residual).map!((e) => e[0] + e[1]).array;

            // 2) MLP block
            x_residual = x;
            x = rmsnorm(x).array;
            x = linear(x, li.mlp_fc1);
            x = x.map!((xi) => xi.relu()).array;
            x = linear(x, li.mlp_fc2);
            x = zip(x, x_residual).map!((e) => e[0] + e[1]).array;
        }

        auto logits = linear(x, lm_head);
        return logits;
    }

    // Let there be Adam, the blessed optimizer and its buffers
    const float learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
    /// first moment buffer
    auto m = new float[params.length]; m[] = 0;
    /// second moment buffer
    auto v = new float[params.length]; v[] = 0;

    // Repeat in sequence
    const num_steps = 1000; /// number of training steps
    foreach(step; 0 .. num_steps)
    {
        // Take single document, tokenize it, surround it with BOS special token on both sides
        auto doc = docs[step % docs.length];
        size_t[] tokens = [BOS];
        foreach(ch; doc)
        {
            auto r = uchars.countUntil(ch);
            enforce(r + 1, "index not found for char: "~ch.to!string);
            tokens ~= r;
        }
        tokens ~= BOS;
        auto n = min(block_size, tokens.length - 1);

    //~ # Forward the token sequence through the model, building up the computation graph all the way to the loss
    //~ keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    //~ losses = []
    //~ for pos_id in range(n):
        //~ token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        //~ logits = gpt(token_id, pos_id, keys, values)
        //~ probs = softmax(logits)
        //~ loss_t = -probs[target_id].log()
        //~ losses.append(loss_t)
    //~ loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    //~ # Backward the loss, calculating the gradients with respect to all model parameters
    //~ loss.backward()

    //~ # Adam optimizer update: update the model parameters based on the corresponding gradients
    //~ lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    //~ for i, p in enumerate(params):
        //~ m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        //~ v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        //~ m_hat = m[i] / (1 - beta1 ** (step + 1))
        //~ v_hat = v[i] / (1 - beta2 ** (step + 1))
        //~ p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        //~ p.grad = 0

    //~ print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
    }

//~ # Inference: may the model babble back to us
//~ temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
//~ print("\n--- inference (new, hallucinated names) ---")
//~ for sample_idx in range(20):
    //~ keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    //~ token_id = BOS
    //~ sample = []
    //~ for pos_id in range(block_size):
        //~ logits = gpt(token_id, pos_id, keys, values)
        //~ probs = softmax([l / temperature for l in logits])
        //~ token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        //~ if token_id == BOS:
            //~ break
        //~ sample.append(uchars[token_id])
    //~ print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
}

float randomGauss(RNG)(ref RNG rng, float std)
{
    // Box-Muller method
    import std.math.exponential: logf = log;

    const u1 = uniform!("(]")(0.0f, 1.0f, rng);
    const u2 = uniform!("(]")(0.0f, 1.0f, rng);
    const z = sqrt(-2.0f * logf(u1)) * cos(2.0f * PI * u2);

    return z * std;
}

auto sumVals(T)(T range) pure => range.fold!((a, b) => a + b);
