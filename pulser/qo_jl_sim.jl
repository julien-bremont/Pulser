using QuantumOptics

function build_operator(op, b, qindex, global_op)
    if global_op
        return sum(build_operator(op, b, q, false) for q in qindex)
    else
        return embed(b, qindex, op)
    end
end

function tensor_basis(b, N)
    if N==1
        return "no"
    else
        return b^N
    end
end

function product_list(list)
    result = 1
    for x in list
        result *= x
    end
    return result
end

function tensor_list(list)
    result = list[1]
    for x in list[2:length(list)]
        result = tensor(result, x)
    end
    return result
end

function build_hamiltonian(terms)
    f = function(t)
        return sum((t * o) + dagger(t * o) for (o, c) in terms) end
    return f
end

function test(op, c)
    f = function(t)
        return c(t) * op end
    return f
end
