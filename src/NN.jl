# fast simple barebones cpu only allocation free dense neural networks 
# use Enzyme for gradients 
module SimpleNN
import Base.zero

function Base.tanh(ret, x)
    ret .= tanh.(x)
    nothing
end

function relu(ret, x)
    ret .= max.(0.0, x)
    nothing
end

function lrelu(ret, x)
    ret .= max.(0.1x, x)
    nothing
end

"""
    softmax function
"""
function softmax(ret, x)
    ret .= exp.(x)
    ss = sum(ret)
    ret .= ret ./ ss
    nothing
end

"""
    make softmax with inv temp β
"""
function softmaxmaker(β)
    function softmax(ret, x)
        ret .= exp.(β * x)
        ss = sum(ret)
        ret .= ret ./ ss
        nothing
    end
end

"""
    Dense layer 
    n_inp: is the size of the input,
    n_nodes: is the size of the output ( or number of nodes in this layer)
    W: weight matrix of size (n_nodes,n_inp)
    b: bias vector (size (n_inp,)
    activation: activation function f, should be able to run f(ret,inp).
"""
struct Dense{T,F<:Function}
    n_inp::Int
    n_nodes::Int
    W::Matrix{T}
    b::Vector{T}
    activation::F
end

"""
    dense layer ,
    f ::activation (should take arguments (ret,inp) and store outputs on `ret`. check `relu` for more details 
    randfn :: random function called randfn(a,b) used to initialize the layers matrix 
"""
function Dense(n_inp, n_nodes, f::Function, randfn::Function=rand)
    Dense(n_inp, n_nodes, randfn(n_nodes, n_inp), randfn(n_nodes), f)
end

"""
    Neural network
    n_inp : no of inputs
    layers : array of Layers
    intermediates : array of Vectors storing intermediate outputs of the layers
"""
struct NN{T,L<:Tuple}
    n_inp::Int
    layers::L # Tuple of Dense
    intermediates::Vector{Vector{T}} # preallocated vectors for output of layers
end

"""
    make an NN , consequent layers must have matching inputs and number of nodes
    (i.e n_nodes of i'th layer == n_inp of i+1th layer) 
    #TODO automate this to be nicer. 
"""
function makeNN(n_inp, layers::Array, T::Type=Float64)
    @assert length(layers) >= 1
    @assert n_inp == layers[1].n_inp
    """ assert consecutive layers match in input and nodes"""
    for i in eachindex(layers)[1:end-1]
        @assert layers[i].n_nodes == layers[i+1].n_inp
    end
    NN(n_inp, Tuple(layers), Vector{T}[zeros(layer.n_nodes) for layer in layers])
end

function makeNN(layers::Array, T::Type=Float64)
    makeNN(layers[1].n_inp, layers, T)
end

"""
    get number of parameters in the nn
"""
function paramlength(nn::NN)
    r = 0
    for l in nn.layers
        r = r + length(l.W)
        r = r + length(l.b)
    end
    return r
end

"""
    get the parameters of the nn flattened in an array
"""
function get_nnparams(nn::NN)
    ret = Float64[]
    for l in nn.layers
        append!(ret, l.W)
        append!(ret, l.b)
    end
    return ret
end

function set_denseparams(d::Dense, arr)
    d.W .= reshape(view(arr, 1:length(d.W)), size(d.W))
    d.b .= view(arr, length(d.W)+1:length(d.W)+1+length(d.b)-1)
end

"""
    set a flattened array of params to nn. (this is type stable)
    Note, This does not error if params is larger than number of params of the nn.
"""
@generated function set_nnparams(nn::NN{T,<:NTuple{N,Any}}, nnparams) where {T,N}
    quote
        i = 1
        Base.Cartesian.@nexprs $N j -> begin
            ll = nn.layers[j]
            set_denseparams(ll, view(nnparams, i:i+length(ll.W)+length(ll.b)-1))
            i = i + length(ll.W) + length(ll.b)
        end
        nothing
    end
end

"""
    returns a similar nn with all 0 params and intermediates
    (use make_zero instead if making shadow for autodiff)
"""
function Base.zero(nn::NN)
    newnn = deepcopy(nn)
    for l in newnn.layers
        l.W .= 0.0
        l.b .= 0.0
    end
    for inter in newnn.intermediates
        inter .= 0.0
    end
    return newnn
end

function zero!(nn)
    for l in nn.layers
        l.W .= 0.0
        l.b .= 0.0
    end
    for inter in nn.intermediates
        inter .= 0
    end
end

"""
    apply dense layer on inp and store the result in out.
    inp : a vector of d.inp size.
    out : a vector of d.nodes size.
    note! uses mul!, `inp` and `out` should not be aliased.
"""
function applydense!(d::Dense, inp, out)
    mul!(out, d.W, inp, 1.0, 0.0)
    out .+= d.b
    d.activation(out, out)
    nothing
end


"""
    apply neural network nn on vector `inp` and store result in `out`
"""
@generated function applyNN!(nn::NN{T,<:NTuple{N,Any}}, inp, out) where {T,N}
    quote
        applydense!(nn.layers[1], inp, nn.intermediates[1])
        Base.Cartesian.@nexprs $(N - 1) j -> begin
            applydense!(nn.layers[j+1], nn.intermediates[j], nn.intermediates[j+1])
        end
        out .= nn.intermediates[end]
        nothing
    end
end

export tanh,relu,lrelu,softmax,softmax,softmaxmaker,Dense,NN,makeNN,paramlength,get_nnparams,set_denseparams,set_nnparams,zero,zero!,applydense,applyNN

end
