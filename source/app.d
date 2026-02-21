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
        //~ def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
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

    static auto rmsnorm(R)(R x) pure
    {
        auto ms = x.map!"a*a".sumVals / x.length;
        auto scale = (ms + 1e-5) ^^ -0.5; //TODO: use float.min_normal instead of 1e-5?
        return x.map!((a) => a * scale);
    }

    auto gpt(in ushort token_id, in ushort pos_id, Value[] keys, Value[] values)
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

        //~ q = linear(x, state_dict[f'layer{li}.attn_wq'])
        //~ k = linear(x, state_dict[f'layer{li}.attn_wk'])
        //~ v = linear(x, state_dict[f'layer{li}.attn_wv'])
        //~ keys[li].append(k)
        //~ values[li].append(v)
        //~ x_attn = []
        //~ for h in range(n_head):
            //~ hs = h * head_dim
            //~ q_h = q[hs:hs+head_dim]
            //~ k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            //~ v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            //~ attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            //~ attn_weights = softmax(attn_logits)
            //~ head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            //~ x_attn.extend(head_out)
        //~ x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        //~ x = [a + b for a, b in zip(x, x_residual)]
        //~ # 2) MLP block
        //~ x_residual = x
        //~ x = rmsnorm(x)
        //~ x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        //~ x = [xi.relu() for xi in x]
        //~ x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        //~ x = [a + b for a, b in zip(x, x_residual)]
        }

    //~ logits = linear(x, state_dict['lm_head'])
    //~ return logits
    }

//~ # Let there be Adam, the blessed optimizer and its buffers
//~ learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
//~ m = [0.0] * len(params) # first moment buffer
//~ v = [0.0] * len(params) # second moment buffer

//~ # Repeat in sequence
//~ num_steps = 1000 # number of training steps
//~ for step in range(num_steps):

    //~ # Take single document, tokenize it, surround it with BOS special token on both sides
    //~ doc = docs[step % len(docs)]
    //~ tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    //~ n = min(block_size, len(tokens) - 1)

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
