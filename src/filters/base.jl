abstract type AbstractFilter end

abstract type SimulationBasedFilter <: AbstractFilter end
abstract type DeterministicFilter <: AbstractFilter end

abstract type SMCFilter <: SimulationBasedFilter end