from pydantic import BaseModel


class NomenclaturesGetBrand(BaseModel):
    nomenclatures: list[str]


class NomenclaturesWithBrands(BaseModel):
    brands: list[str]
