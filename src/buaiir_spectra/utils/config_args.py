from dataclasses import dataclass

@dataclass
class ConfigArgs:
    """
    Sets the configuration paremters 

    Arg:
        t_week: desired week to be loaded
        t_disease_class: desired class to be loaded
        t_plant_type: desired plant type to be loaded

    """
    t_week: int | None = None
    t_disease_class: str | None = None
    t_plant_type: str | None = None

    # allowed diseases for each plant
    VALID_CLASSES = {
        'C': {'HLT', 'CBB', 'CMD'},
        'B': {'HLT', 'BRD', 'BLB'},
        'M': {'HLT', 'MSV', 'MLN'}
    }

    def __post_init__(self):
        # Preprocess string fields
        if self.t_plant_type is not None:
            self.t_plant_type = (
                str(self.t_plant_type)
                .strip()
                .upper()
            )

        if self.t_disease_class is not None:
            self.t_disease_class = (
                str(self.t_disease_class)
                .strip()
                .upper()
            )
        
        # Preprocess integer field
        if self.t_week is not None:
            try:
                self.t_week = int(self.t_week)
            except (ValueError, TypeError):
                raise ValueError(
                    "t_week must be convertible to int"
                )

        # valid plant type
        if self.t_plant_type is not None:
            if self.t_plant_type not in self.VALID_CLASSES:
                raise ValueError(
                    f"t_plant_type must be one of ",
                    f"{list(self.VALID_CLASSES.keys())}"
                )
            
        # validate disease class
        if self.t_disease_class is not None:
            # disease class requires plant type
            if self.t_plant_type is None:
                raise ValueError(
                    "t_plant_type cannot be None when",
                    "t_disease_class is provided"
                )
            
            valid_diseases = self.VALID_CLASSES[self.t_plant_type]
            
            if self.t_disease_class not in valid_diseases:
                raise ValueError(
                    f"For plant type `{self.t_plant_type}`,"
                    f"t_disease_class may be one of ",
                    f"{list(valid_diseases)}"
                )


if __name__ == '__main__':

    # valid 
    cf0 = ConfigArgs(t_disease_class='CMD', t_plant_type='C')

    print(cf0.t_disease_class, cf0.t_plant_type)


    