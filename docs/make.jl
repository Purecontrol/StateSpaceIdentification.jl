using Documenter
using StateSpaceIdentification

DocMeta.setdocmeta!(StateSpaceIdentification,
                    :DocTestSetup,
                    :(using StateSpaceIdentification;),
                    recursive = true)

makedocs(
    sitename = "StateSpaceIdentification",
    format = Documenter.HTML(),
    modules = [StateSpaceIdentification],
    remotes = nothing
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#deploydocs(
#    repo = "<repository url>"
#)
