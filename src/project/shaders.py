# shaders.py

vertex_shader_code = """
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
uniform mat4 projection;
uniform mat4 modelview;
out vec2 fragTexCoord;

void main() {
    gl_Position = projection * modelview * vec4(position, 1.0);
    fragTexCoord = texCoord;
}
"""

fragment_shader_code = """
#version 330
in vec2 fragTexCoord;
uniform sampler2D tex;
out vec4 outColor;

void main() {
    vec4 texColor = texture(tex, fragTexCoord);
    outColor = texColor;
    // Debug: output pure red if you don't see anything
    // outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""
