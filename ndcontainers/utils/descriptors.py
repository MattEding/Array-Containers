class FlagDescriptor:
    def __get__(self, instance, owner):
        return getattr(instance._flag_array.flags, self.name)

    def __set__(self, instance, value):
        setattr(instance._flag_array.flags, self.name, value)
        for arr in instance._array_refs:
            setattr(arr.flags, self.name, value)
    
    def __set_name__(self, owner, name):
        self.name = name
