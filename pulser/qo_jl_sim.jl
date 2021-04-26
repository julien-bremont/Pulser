using QuantumOptics

function build_operator(op, b, qindex, global_op)
    if global_op
        return Base.sum([build_operator(op, b, [q], false) for q in qindex])
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

function product_list(list, b)
    result = identityoperator(b)
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

function convert_ham(f)
    function h(t, psi)
        return f(t)
    end
end

function test(op, c)
    f = function(t)
        return c(t) * op end
    return f
end
