using QuantumOptics

function tensor_basis(b, N)
    return b^N
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
