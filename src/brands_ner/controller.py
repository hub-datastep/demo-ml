from fastapi import APIRouter, HTTPException, status

from brands_ner.model import brand_ner_model
from brands_ner.scheme import NomenclaturesGetBrand, NomenclaturesWithBrands

brands_ner_router = APIRouter()


@brands_ner_router.post("", response_model=NomenclaturesWithBrands)
async def get_brands(body: NomenclaturesGetBrand):
    try:
        brands = brand_ner_model.get_all_ner_brands(body.nomenclatures)
        print(brands)
        return NomenclaturesWithBrands(brands=brands)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
