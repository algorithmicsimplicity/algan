{{ name | escape | underline}}

Qualified name: ``{{ fullname | escape }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:

{# The tables list only what this class defines itself. The autoclass above
   documents nothing else (it has no :inherited-members:), so a row for an
   inherited method or attribute linked to nowhere; the base classes that
   :show-inheritance: names are where those are documented. #}
   {% block methods %}
   {%- set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list %}
   {%- if own_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      {% for item in own_methods %}
      ~{{ name }}.{{ item }}
      {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes %}
   {%- set own_attributes = attributes | reject("in", inherited_members) | list %}
   {%- if own_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
     {% for item in own_attributes %}
     ~{{ name }}.{{ item }}
     {%- endfor %}
   {%- endif %}
   {% endblock %}
