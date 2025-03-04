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
out vec4 outColor;
uniform sampler2D videoTexture;
void main() {
    outColor = texture(videoTexture, fragTexCoord);
}
"""
