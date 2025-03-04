# shaders.py

vertex_shader_code = """
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
uniform mat4 projection;
uniform mat4 modelview;
out vec2 fragTexCoord;

void main() {
    vec4 pos = projection * modelview * vec4(position, 1.0);
    gl_Position = pos;
    fragTexCoord = texCoord;
}
"""

fragment_shader_code = """
#version 330
in vec2 fragTexCoord;
uniform sampler2D tex;
out vec4 outColor;

void main() {
    outColor = texture(tex, fragTexCoord);
}
"""
