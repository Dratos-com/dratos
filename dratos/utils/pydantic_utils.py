from pydantic import BaseModel
from typing import List, Tuple, Union, Type, get_origin, get_args, Optional, Any
from pydantic import ValidationError


# Tuple: (model, associated data)
NestedModelData = Tuple[Union[BaseModel, Type[BaseModel]], dict]

def extract_nested_models_with_data(model: Union[BaseModel, Type[BaseModel]], data: dict) -> List[NestedModelData]:
    results = []
    
    # Helper function for class extraction
    def extract_from_class(model_cls: Type[BaseModel], data: dict):
        for field_name, field in model_cls.__fields__.items():
            # Get the field type (try type_ then annotation)
            field_type = getattr(field, 'type_', None) or getattr(field, 'annotation', None)
            if field_name not in data:
                continue

            # Direct nested model
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                field_data = data[field_name]
                if isinstance(field_data, dict):
                    results.append((field_type, field_data))
                    extract_from_class(field_type, field_data)
            else:
                # Check if it's a container like a list of models
                outer_type = getattr(field, 'outer_type_', None) or field_type
                if get_origin(outer_type) is list:
                    for arg in get_args(outer_type):
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            field_list = data.get(field_name, [])
                            if isinstance(field_list, list):
                                for item in field_list:
                                    if isinstance(item, dict):
                                        results.append((arg, item))
                                        extract_from_class(arg, item)
    
    # Helper function for instance extraction
    def extract_from_instance(instance: BaseModel, data: dict):
        for field_name in instance.__fields__:
            if field_name not in data:
                continue
            value = getattr(instance, field_name)
            field_data = data[field_name]
            if isinstance(value, BaseModel) and isinstance(field_data, dict):
                results.append((value, field_data))
                extract_from_instance(value, field_data)
            elif isinstance(value, list) and isinstance(field_data, list):
                for idx, item in enumerate(value):
                    if isinstance(item, BaseModel) and idx < len(field_data) and isinstance(field_data[idx], dict):
                        results.append((item, field_data[idx]))
                        extract_from_instance(item, field_data[idx])
    
    if isinstance(model, type):
        extract_from_class(model, data)
    else:
        extract_from_instance(model, data)
    
    return results

def largest_invalid_model(tuple_pairs: List[NestedModelData]) -> Optional[NestedModelData]:
    invalid = []
    for model, data in tuple_pairs:
        # Get the model class even if model is an instance
        model_cls = model if isinstance(model, type) else type(model)
        try:
            # Try validating the data
            model_cls.model_validate(data)
        except ValidationError:
            # If validation fails, store the tuple
            invalid.append((model, data))
    if not invalid:
        return None
    # Return the tuple with the largest dict (largest by number of keys)
    return max(invalid, key=lambda pair: len(pair[1]))

def remove_invalid_data(data: Union[dict, list], invalid_data: dict) -> Union[dict, list]:
    """
    Recursively traverse 'data' and remove any dict that equals 'invalid_data'.
    Works for nested dictionaries and lists.
    """
    if isinstance(data, dict):
        # Create a copy of keys to remove after iteration.
        keys_to_remove = []
        for key, value in data.items():
            if isinstance(value, dict) and value == invalid_data:
                keys_to_remove.append(key)
            else:
                data[key] = remove_invalid_data(value, invalid_data)
        for key in keys_to_remove:
            del data[key]
        return data
    elif isinstance(data, list):
        # Rebuild the list without the invalid_data dict.
        new_list = []
        for item in data:
            if isinstance(item, dict) and item == invalid_data:
                continue
            new_list.append(remove_invalid_data(item, invalid_data))
        return new_list
    else:
        return data

def recursive_model_validate(model: BaseModel, data: dict) -> BaseModel:
    """
    Recursively validate a dictionary against a Pydantic model.
    """
    tuple_pairs = extract_nested_models_with_data(model, data)
    invalid_model = largest_invalid_model(tuple_pairs)
    if invalid_model:
        invalid_data = invalid_model[1]
        return remove_invalid_data(data, invalid_data), invalid_model
        
    else:
        return model.model_validate(data), None


def recursive_merge(value1: Any, value2: Any) -> Any:
    # If both values are Pydantic models, merge their dict representations.
    if isinstance(value1, BaseModel) and isinstance(value2, BaseModel):
        merged = recursive_merge(value1.dict(), value2.dict())
        return value1.__class__(**merged)
    # Merge dictionaries key by key.
    elif isinstance(value1, dict) and isinstance(value2, dict):
        merged = {}
        all_keys = set(value1.keys()).union(value2.keys())
        for key in all_keys:
            if key in value1 and key in value2:
                merged[key] = recursive_merge(value1[key], value2[key])
            elif key in value1:
                merged[key] = value1[key]
            else:
                merged[key] = value2[key]
        return merged
    # If both are lists, concatenate them.
    elif isinstance(value1, list) and isinstance(value2, list):
        return value1 + value2
    # Otherwise, prefer value2 if it exists.
    else:
        return value2 if value2 is not None else value1

def merge_pydantic_models(model1: BaseModel, model2: BaseModel) -> BaseModel:
    merged_data = recursive_merge(model1.dict(), model2.dict())
    return model1.__class__(**merged_data)