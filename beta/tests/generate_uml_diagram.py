from pydantic_to_diagram import generate_diagram


from .. import beta

import erdantic as erd


diagram = erd.create(beta)
diagram
