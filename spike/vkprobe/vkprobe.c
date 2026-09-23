// vkprobe: standalone spike for the PyroWave integration. Answers the two
// unknowns before any wiring:
//   1. Can Vulkan import an NVIDIA-tiled (DRM format modifier) dma-buf as a
//      sampleable VkImage? (We fabricate a scanout-like tiled dma-buf with
//      GBM, since we can't GETFB2 without CAP_SYS_ADMIN.)
//   2. Can a Vulkan HOST_VISIBLE|HOST_COHERENT buffer be registered as an
//      RDMA MR (ibv_reg_mr) for the existing staging-slot path?
// Also reports whether the device exposes PyroWave's required features.
//
// Build:  gcc -O2 -o vkprobe vkprobe.c -I$VKHDR -lvulkan -lgbm -libverbs
// Run:    ./vkprobe [renderNode]

#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <gbm.h>
#include <infiniband/verbs.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VK_CHECK(x)                                                            \
	do {                                                                   \
		VkResult _r = (x);                                             \
		if (_r != VK_SUCCESS) {                                        \
			fprintf(stderr, "FATAL: %s -> VkResult %d\n", #x, _r); \
			exit(1);                                               \
		}                                                              \
	} while (0)

#define OK "OK  "
#define NO "FAIL"

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;

static int g_has_dmabuf;
static int g_has_host;
static int g_drm_node_open;

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetDeviceProcAddr gdpa;

static PFN_vkVoidFunction I(const char *n) { return gipa(inst, n); }
static PFN_vkVoidFunction D(const char *n) { return gdpa(dev, n); }

static const char *vkres(VkResult r) {
	static char b[32];
	snprintf(b, sizeof b, "VkResult(%d)", r);
	return b;
}

static uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(phys, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
			return i;
	fprintf(stderr, "FATAL: no memory type for bits=0x%x want=0x%x\n", bits, want);
	exit(1);
}

// ---------------------------------------------------------------- instance --

static int has_ext(const VkExtensionProperties *p, uint32_t n, const char *name) {
	for (uint32_t i = 0; i < n; i++)
		if (!strcmp(p[i].extensionName, name)) return 1;
	return 0;
}

static void create_instance(void) {
	VkApplicationInfo ai = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.apiVersion = VK_API_VERSION_1_3,
	};
	VkInstanceCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &ai,
	};
	VK_CHECK(vkCreateInstance(&ci, NULL, &inst));
	gipa = vkGetInstanceProcAddr;
}

static void pick_device(void) {
	uint32_t n = 0;
	vkEnumeratePhysicalDevices(inst, &n, NULL);
	VkPhysicalDevice *pds = calloc(n, sizeof(*pds));
	vkEnumeratePhysicalDevices(inst, &n, pds);

	const char *need[] = {
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
		VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
	};
	for (uint32_t i = 0; i < n; i++) {
		VkPhysicalDeviceProperties p;
		vkGetPhysicalDeviceProperties(pds[i], &p);
		uint32_t en = 0;
		vkEnumerateDeviceExtensionProperties(pds[i], NULL, &en, NULL);
		VkExtensionProperties *ep = calloc(en, sizeof(*ep));
		vkEnumerateDeviceExtensionProperties(pds[i], NULL, &en, ep);
		printf("[probe] device[%u] %s (%u extensions)\n", i, p.deviceName, en);
		int dump_all = getenv("VKPROBE_ALL") != NULL;
		for (uint32_t e = 0; e < en; e++)
			if (dump_all || strstr(ep[e].extensionName, "external") || strstr(ep[e].extensionName, "drm") ||
			    strstr(ep[e].extensionName, "dma") || strstr(ep[e].extensionName, "memory"))
				printf("[probe]     %s\n", ep[e].extensionName);
		int ok = 1;
		for (unsigned k = 0; k < sizeof(need) / sizeof(need[0]); k++)
			if (!has_ext(ep, en, need[k])) ok = 0, fprintf(stderr, "  missing %s\n", need[k]);
		g_has_dmabuf = has_ext(ep, en, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
		g_has_host = has_ext(ep, en, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
		free(ep);
		if (!ok) continue;
		phys = pds[i];
		printf("[device] %s (apiVersion %u.%u.%u, driver 0x%x)\n",
		       p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
		       VK_VERSION_PATCH(p.apiVersion), p.driverVersion);
		break;
	}
	free(pds);
	if (phys == VK_NULL_HANDLE) { fprintf(stderr, "no suitable device\n"); exit(1); }
}

static void create_device(void) {
	uint32_t qn = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, NULL);
	VkQueueFamilyProperties *q = calloc(qn, sizeof(*q));
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, q);
	qfam = 0;
	for (uint32_t i = 0; i < qn; i++)
		if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio,
	};
	const char *want[] = {
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
		VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
		VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
		VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
	};
	uint32_t en = 0;
	vkEnumerateDeviceExtensionProperties(phys, NULL, &en, NULL);
	VkExtensionProperties *ep = calloc(en, sizeof(*ep));
	vkEnumerateDeviceExtensionProperties(phys, NULL, &en, ep);
	const char *ext[8];
	uint32_t ne = 0;
	for (unsigned k = 0; k < sizeof(want) / sizeof(want[0]); k++)
		if (has_ext(ep, en, want[k])) ext[ne++] = want[k];
	free(ep);
	VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	VkPhysicalDeviceVulkan12Features f12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f11,
		.storageBuffer8BitAccess = 1,
	};
	VkPhysicalDeviceVulkan13Features f13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &f12,
		.subgroupSizeControl = 1,
	};
	VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13 };
	f2.features.shaderInt16 = 1;
	VkDeviceCreateInfo dci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
		.queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
		.enabledExtensionCount = ne, .ppEnabledExtensionNames = ext,
	};
	VK_CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
	gdpa = (PFN_vkGetDeviceProcAddr)I("vkGetDeviceProcAddr");
	vkGetDeviceQueue(dev, qfam, 0, &queue);
	printf("[queue] family %u (compute)\n", qfam);
}

// --------------------------------------------------------------- features ---

static void report_features(void) {
	VkPhysicalDeviceSubgroupProperties sg = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
	VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &f12 };
	VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13 };
	VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg };
	vkGetPhysicalDeviceFeatures2(phys, &f2);
	vkGetPhysicalDeviceProperties2(phys, &p2);
	int shader_int16 = f2.features.shaderInt16;

	VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
		VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
		VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
		VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
	printf("\n[pyrowave features]\n");
	printf("  %s subgroup ops (have 0x%x, need 0x%x)\n", (sg.supportedOperations & need) == need ? OK : NO, sg.supportedOperations, need);
	printf("  %s subgroup size control\n", f13.subgroupSizeControl ? OK : NO);
	printf("  %s shaderInt16\n", shader_int16 ? OK : NO);
	printf("  %s storageBuffer8BitAccess\n", f12.storageBuffer8BitAccess ? OK : NO);
	printf("      subgroupSize=%u\n", sg.subgroupSize);
}

// ------------------------------------------------- DRM format modifiers -----

static VkFormat FMT = VK_FORMAT_B8G8R8A8_UNORM; // XR24/AR24
static VkImageUsageFlags USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

static int modifier_supported(uint64_t m, char *why, size_t n) {
	PFN_vkGetPhysicalDeviceImageFormatProperties2 f =
		(PFN_vkGetPhysicalDeviceImageFormatProperties2)I("vkGetPhysicalDeviceImageFormatProperties2");
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT mi = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
		.drmFormatModifier = m, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkPhysicalDeviceExternalImageFormatInfo ei = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
		.pNext = &mi, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkPhysicalDeviceImageFormatInfo2 fi = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .pNext = &ei,
		.format = FMT, .type = VK_IMAGE_TYPE_2D,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT, .usage = USAGE,
	};
	VkExternalImageFormatProperties ep = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };
	VkImageFormatProperties2 ip = { .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, .pNext = &ep };
	VkResult r = f(phys, &fi, &ip);
	if (r != VK_SUCCESS) { snprintf(why, n, "%s", vkres(r)); return 0; }
	if (!(ep.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
		snprintf(why, n, "not importable"); return 0;
	}
	snprintf(why, n, "sampleable+importable"); return 1;
}

static void list_modifiers(uint64_t *out_best) {
	PFN_vkGetPhysicalDeviceFormatProperties2 f =
		(PFN_vkGetPhysicalDeviceFormatProperties2)I("vkGetPhysicalDeviceFormatProperties2");
	VkDrmFormatModifierPropertiesListEXT ml = { .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT };
	VkFormatProperties2 fp = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &ml };
	f(phys, FMT, &fp);
	uint32_t n = ml.drmFormatModifierCount;
	VkDrmFormatModifierPropertiesEXT *mp = calloc(n, sizeof(*mp));
	ml.pDrmFormatModifierProperties = mp;
	f(phys, FMT, &fp);
	printf("\n[vulkan DRM modifiers for %s] %u total\n", "B8G8R8A8_UNORM", n);
	*out_best = 0;
	for (uint32_t i = 0; i < n && i < 12; i++) {
		char why[64];
		int ok = modifier_supported(mp[i].drmFormatModifier, why, sizeof why);
		printf("  %s modifier 0x%016llx (%u plane%s): tiling=0x%x sampled=%d | %s\n",
		       (ok ? OK : NO), (unsigned long long)mp[i].drmFormatModifier,
		       mp[i].drmFormatModifierPlaneCount, mp[i].drmFormatModifierPlaneCount > 1 ? "s" : "",
		       mp[i].drmFormatModifierTilingFeatures,
		       (mp[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0,
		       why);
		if (ok && *out_best == 0) *out_best = mp[i].drmFormatModifier;
	}
	if (n > 12) printf("  ... (%u more)\n", n - 12);
	free(mp);
}

// ---------------------------------------------------------- GBM dma-buf -----

static int gbm_make(int node, uint32_t w, uint32_t h, uint64_t *mod, uint32_t *stride,
                    uint32_t *offset, int *out_render_fd) {
	int fd = open(node ? "/dev/dri/renderD128" : "/dev/dri/card0", O_RDWR);
	if (fd < 0) { perror("open drm node"); return -1; }
	struct gbm_device *g = gbm_create_device(fd);
	if (!g) { fprintf(stderr, "gbm_create_device failed\n"); return -1; }
	// legacy create (no modifier list): on this stack it yields the
	// block-linear scanout modifier, which is what KMS scanouts use.
	struct gbm_bo *bo = gbm_bo_create(g, w, h, GBM_FORMAT_ARGB8888,
	                                  GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
	if (!bo) { fprintf(stderr, "gbm_bo_create failed\n"); return -1; }
	*mod = gbm_bo_get_modifier(bo);
	*stride = gbm_bo_get_stride(bo);
	*offset = gbm_bo_get_offset(bo, 0);
	int d = gbm_bo_get_fd(bo);
	printf("\n[gbm dma-buf] %ux%u ARGB8888 modifier=0x%016llx stride=%u offset=%u fd=%d\n",
	       w, h, (unsigned long long)*mod, *stride, *offset, d);
	*out_render_fd = fd;
	return d;
}

// ------------------------------------------------------------- import test --

static VkCommandBuffer one_shot_begin(VkCommandPool pool) {
	VkCommandBufferAllocateInfo ai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
	};
	VkCommandBuffer cb;
	VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
	VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	VK_CHECK(vkBeginCommandBuffer(cb, &bi));
	return cb;
}

static void one_shot_end(VkCommandPool pool, VkCommandBuffer cb) {
	VK_CHECK(vkEndCommandBuffer(cb));
	VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
	VK_CHECK(vkQueueWaitIdle(queue));
	vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void import_and_roundtrip(int dmabuf, uint64_t mod, uint32_t stride, uint32_t offset,
                                 uint32_t w, uint32_t h) {
	PFN_vkGetImageMemoryRequirements2 req2 = (PFN_vkGetImageMemoryRequirements2)D("vkGetImageMemoryRequirements2");
	PFN_vkBindImageMemory2 bind2 = (PFN_vkBindImageMemory2)D("vkBindImageMemory2");
	PFN_vkGetImageDrmFormatModifierPropertiesEXT modprops =
		(PFN_vkGetImageDrmFormatModifierPropertiesEXT)D("vkGetImageDrmFormatModifierPropertiesEXT");

	VkSubresourceLayout plane = { .offset = offset, .rowPitch = stride };
	VkImageDrmFormatModifierExplicitCreateInfoEXT mc = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
		.drmFormatModifier = mod, .drmFormatModifierPlaneCount = 1, .pPlaneLayouts = &plane,
	};
	VkExternalMemoryImageCreateInfo em = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
		.pNext = &mc, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &em,
		.imageType = VK_IMAGE_TYPE_2D, .format = FMT, .extent = { w, h, 1 },
		.mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT, .usage = USAGE,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkImage img;
	VkResult r = vkCreateImage(dev, &ici, NULL, &img);
	printf("\n[import] vkCreateImage(modifier 0x%016llx, %ux%u, SAMPLED|TRANSFER): %s\n",
	       (unsigned long long)mod, w, h, r == VK_SUCCESS ? OK : vkres(r));
	if (r != VK_SUCCESS) return;

	VkImageDrmFormatModifierPropertiesEXT actual = { .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT };
	if (modprops && modprops(dev, img, &actual) == VK_SUCCESS)
		printf("[import]   actual modifier on image: 0x%016llx\n", (unsigned long long)actual.drmFormatModifier);
	else
		printf("[import]   (vkGetImageDrmFormatModifierPropertiesEXT unavailable)\n");

	VkMemoryRequirements mr;
	VkImageMemoryRequirementsInfo2 ri = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = img };
	VkMemoryRequirements2 mr2 = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
	req2(dev, &ri, &mr2);
	mr = mr2.memoryRequirements;

	VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = img };
	VkImportMemoryFdInfoKHR imp = {
		.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .pNext = &ded,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = dmabuf,
	};
	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
		.allocationSize = mr.size,
		.memoryTypeIndex = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
	};
	VkDeviceMemory mem;
	r = vkAllocateMemory(dev, &mai, NULL, &mem);
	printf("[import] vkAllocateMemory(import dma-buf fd): %s\n", r == VK_SUCCESS ? OK : vkres(r));
	if (r != VK_SUCCESS) { vkDestroyImage(dev, img, NULL); return; }

	VkBindImageMemoryInfo bi = { .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, .image = img, .memory = mem, .memoryOffset = 0 };
	r = bind2(dev, 1, &bi);
	printf("[import] vkBindImageMemory2: %s\n", r == VK_SUCCESS ? OK : vkres(r));
	if (r != VK_SUCCESS) return;

	// GPU round trip: clear the imported image, copy it to a host buffer,
	// and verify the CPU sees the cleared color (proves the import is a
	// usable, coherent image; sampling is covered by the modifier query).
	VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam };
	VkCommandPool pool;
	VK_CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));

	VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * 4,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkBuffer buf;
	VK_CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
	VkMemoryRequirements bmr;
	vkGetBufferMemoryRequirements(dev, buf, &bmr);
	VkMemoryAllocateInfo bmai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = bmr.size,
		.memoryTypeIndex = find_mem_type(bmr.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
	};
	VkDeviceMemory bmem;
	VK_CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
	VK_CHECK(vkBindBufferMemory(dev, buf, bmem, 0));

	VkCommandBuffer cb = one_shot_begin(pool);
	VkImageMemoryBarrier b = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
	};
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
	VkClearColorValue col = { .float32 = { 0.25f, 0.5f, 0.75f, 1.0f } };
	vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &col, 1, &b.subresourceRange);
	VkImageMemoryBarrier b2 = b;
	b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	b2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b2);
	VkBufferImageCopy region = {
		.bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { w, h, 1 },
	};
	vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
	one_shot_end(pool, cb);

	void *p = NULL;
	VK_CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, &p));
	uint32_t px = ((uint32_t *)p)[0];
	// BGRA clear (0.25,0.5,0.75,1) -> B=64,G=128,R=191,A=255
	uint32_t want = (255u << 24) | (191u << 16) | (128u << 8) | 64u;
	printf("[import] GPU clear->copy->CPU readback: %s (got 0x%08x want 0x%08x)\n",
	       px == want ? OK : "MISMATCH", px, want);
	vkUnmapMemory(dev, bmem);
	vkDestroyBuffer(dev, buf, NULL);
	vkFreeMemory(dev, bmem, NULL);
	vkDestroyCommandPool(dev, pool, NULL);
}

// ------------------------------------------------------------- RDMA test ----

// Whether the process can open the DRM render node. NVIDIA gates
// VK_EXT_external_memory_dma_buf on nvidia-drm access, so a sandbox without
// it will not see the extension even though the driver supports it.
static void drm_gate_report(void) {
	int fd = open("/dev/dri/renderD128", O_RDWR);
	g_drm_node_open = fd >= 0;
	printf("\n[gate] open /dev/dri/renderD128: %s\n", g_drm_node_open ? OK : "denied");
	if (fd >= 0) close(fd);
	printf("[gate] VK_EXT_external_memory_dma_buf advertised: %s\n", g_has_dmabuf ? OK : "no");
	printf("[gate] VK_EXT_external_memory_host advertised:   %s\n", g_has_host ? OK : "no");
	if (!g_has_dmabuf)
		printf("[gate] NVIDIA gates dma_buf on DRM render-node access; run on the\n"
		       "       lab host (sudo / render group) for a valid answer.\n");
}

// Fallback for driving PyroWave without dma-buf: import a host pointer as
// Vulkan memory (VK_EXT_external_memory_host) and write a linear image from it.
static void host_import_test(void) {
	if (!g_has_host) { printf("\n[host-import] not advertised\n"); return; }
	PFN_vkGetMemoryHostPointerPropertiesEXT gh =
		(PFN_vkGetMemoryHostPointerPropertiesEXT)D("vkGetMemoryHostPointerPropertiesEXT");
	if (!gh) { printf("\n[host-import] entry point missing\n"); return; }

	VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp =
		{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
	VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp };
	vkGetPhysicalDeviceProperties2(phys, &p2);
	size_t align = hp.minImportedHostPointerAlignment ? hp.minImportedHostPointerAlignment : 4096;
	printf("\n[host-import] minImportedHostPointerAlignment=%zu\n", align);

	// --- buffer import ---
	size_t bsize = align * 256;
	void *bp = NULL;
	if (posix_memalign(&bp, align, bsize)) { printf("[host-import] alloc failed\n"); return; }
	memset(bp, 0, bsize);
	VkMemoryHostPointerPropertiesEXT mhp = { .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
	VkResult r = gh(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, bp, &mhp);
	printf("[host-import] vkGetMemoryHostPointerPropertiesEXT: %s\n", r == VK_SUCCESS ? OK : vkres(r));
	if (r != VK_SUCCESS) { free(bp); return; }

	VkExternalMemoryBufferCreateInfo emb = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	};
	VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &emb, .size = bsize,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkBuffer buf;
	VK_CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(dev, buf, &mr);
	uint32_t mt = find_mem_type(mhp.memoryTypeBits & mr.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = buf };
	VkImportMemoryHostPointerInfoEXT imp = {
		.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, .pNext = &ded,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = bp,
	};
	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
		.allocationSize = mr.size, .memoryTypeIndex = mt,
	};
	VkDeviceMemory bmem;
	r = vkAllocateMemory(dev, &mai, NULL, &bmem);
	printf("[host-import] buffer import: %s\n", r == VK_SUCCESS ? OK : vkres(r));
	if (r == VK_SUCCESS) {
		VK_CHECK(vkBindBufferMemory(dev, buf, bmem, 0));
		VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam };
		VkCommandPool pool;
		VK_CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
		VkCommandBuffer cb = one_shot_begin(pool);
		vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0xCAFED00Du);
		one_shot_end(pool, cb);
		printf("[host-import] GPU vkCmdFillBuffer -> CPU sees 0x%08x: %s\n",
		       ((uint32_t *)bp)[0], ((uint32_t *)bp)[0] == 0xCAFED00Du ? OK : "MISMATCH");
		vkDestroyCommandPool(dev, pool, NULL);
	}

	// --- linear R8 image import (PyroWave plane shape) ---
	VkExternalMemoryImageCreateInfo emi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	};
	VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi,
		.imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8_UNORM, .extent = { 256, 256, 1 },
		.mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_LINEAR,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkImage img;
	r = vkCreateImage(dev, &ici, NULL, &img);
	printf("[host-import] linear R8 VkImage: %s\n", r == VK_SUCCESS ? OK : vkres(r));
	if (r == VK_SUCCESS) {
		VkMemoryRequirements imr;
		vkGetImageMemoryRequirements(dev, img, &imr);
		size_t isize = ((imr.size + align - 1) / align) * align;
		void *ip = NULL;
		if (!posix_memalign(&ip, align, isize)) {
			memset(ip, 0, isize);
			VkImageSubresource isr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
			VkSubresourceLayout sl;
			vkGetImageSubresourceLayout(dev, img, &isr, &sl);
			for (uint32_t y = 0; y < 256; y++)
				memset((uint8_t *)ip + sl.offset + y * sl.rowPitch, (uint8_t)y, 256);
			VkMemoryDedicatedAllocateInfo ided = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = img };
			VkImportMemoryHostPointerInfoEXT iimp = {
				.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, .pNext = &ided,
				.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = ip,
			};
			VkMemoryAllocateInfo imai = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &iimp,
				.allocationSize = imr.size,
				.memoryTypeIndex = find_mem_type(mhp.memoryTypeBits & imr.memoryTypeBits,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
			};
			VkDeviceMemory imem;
			r = vkAllocateMemory(dev, &imai, NULL, &imem);
			printf("[host-import] image memory import: %s (rowPitch=%llu)\n",
			       r == VK_SUCCESS ? OK : vkres(r), (unsigned long long)sl.rowPitch);
			if (r == VK_SUCCESS) {
				VK_CHECK(vkBindImageMemory(dev, img, imem, 0));
				// copy image -> device buffer and verify the host-written rows
				VkBufferCreateInfo dci = {
					.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 256 * 256,
					.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				};
				VkBuffer dbuf;
				VK_CHECK(vkCreateBuffer(dev, &dci, NULL, &dbuf));
				VkMemoryRequirements dmr;
				vkGetBufferMemoryRequirements(dev, dbuf, &dmr);
				VkMemoryAllocateInfo dmai = {
					.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = dmr.size,
					.memoryTypeIndex = find_mem_type(dmr.memoryTypeBits,
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
				};
				VkDeviceMemory dmem;
				VK_CHECK(vkAllocateMemory(dev, &dmai, NULL, &dmem));
				VK_CHECK(vkBindBufferMemory(dev, dbuf, dmem, 0));
				VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam };
				VkCommandPool pool;
				VK_CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
				VkCommandBuffer cb = one_shot_begin(pool);
				VkImageMemoryBarrier b = {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
					.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
				};
				vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
				VkBufferImageCopy reg = {
					.bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
					.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
					.imageExtent = { 256, 256, 1 },
				};
				vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dbuf, 1, &reg);
				one_shot_end(pool, cb);
				void *dp = NULL;
				VK_CHECK(vkMapMemory(dev, dmem, 0, VK_WHOLE_SIZE, 0, &dp));
				uint8_t got = ((uint8_t *)dp)[5 * 256 + 3];
				printf("[host-import] image copy-out: row5 val=%u (want 5): %s\n",
				       got, got == 5 ? OK : "MISMATCH");
				vkUnmapMemory(dev, dmem);
				vkDestroyCommandPool(dev, pool, NULL);
			}
			free(ip);
		}
	}
	free(bp);
}

static void rdma_test(void) {
	// Vulkan host-visible buffer, GPU-filled, then CPU-verified + registered.
	VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 1u << 20,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkBuffer buf;
	VK_CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(dev, buf, &mr);
	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
		.memoryTypeIndex = find_mem_type(mr.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
	};
	VkDeviceMemory mem;
	VK_CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
	VK_CHECK(vkBindBufferMemory(dev, buf, mem, 0));

	VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam };
	VkCommandPool pool;
	VK_CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
	VkCommandBuffer cb = one_shot_begin(pool);
	vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0xC0FFEE42u);
	one_shot_end(pool, cb);

	void *p = NULL;
	VK_CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p));
	uint32_t v = ((uint32_t *)p)[0];
	printf("\n[rdma] VkBuffer HOST_VISIBLE|HOST_COHERENT, GPU vkCmdFillBuffer -> CPU sees 0x%08x: %s\n",
	       v, v == 0xC0FFEE42u ? OK : "MISMATCH");

	int ndev = 0;
	struct ibv_device **list = ibv_get_device_list(&ndev);
	printf("[rdma] ibv_get_device_list: %d device(s)\n", ndev);
	if (!list || ndev < 1 || !list[0]) { printf("[rdma] no RDMA devices\n"); return; }
	printf("[rdma] first device: %s\n", ibv_get_device_name(list[0]));
	struct ibv_context *ctx = ibv_open_device(list[0]);
	if (!ctx) { printf("[rdma] ibv_open_device failed\n"); return; }
	struct ibv_pd *pd = ibv_alloc_pd(ctx);
	if (!pd) { printf("[rdma] ibv_alloc_pd failed\n"); return; }
	struct ibv_mr *m = ibv_reg_mr(pd, p, mr.size,
		IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
	printf("[rdma] ibv_reg_mr(Vulkan host pointer, %zu bytes): %s\n", mr.size,
	       m ? OK : "FAIL");
	if (m)
		printf("[rdma]   lkey=0x%x rkey=0x%x\n", m->lkey, m->rkey);

	// best effort: export the memory as a dma-buf and register that too
	PFN_vkGetMemoryFdKHR getfd = (PFN_vkGetMemoryFdKHR)D("vkGetMemoryFdKHR");
	if (getfd) {
		VkMemoryGetFdInfoKHR gi = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
			.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};
		int fd = -1;
		VkResult r = getfd(dev, &gi, &fd);
		printf("[rdma] vkGetMemoryFdKHR(dma-buf): %s", r == VK_SUCCESS ? OK : vkres(r));
		if (r == VK_SUCCESS) {
			struct ibv_mr *dm = ibv_reg_dmabuf_mr(pd, 0, mr.size, (uint64_t)(uintptr_t)p, fd,
				IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
			printf(" | ibv_reg_dmabuf_mr: %s\n", dm ? OK : "FAIL");
			if (dm) ibv_dereg_mr(dm);
		} else printf("\n");
	} else {
		printf("[rdma] vkGetMemoryFdKHR unavailable\n");
	}
	if (m) ibv_dereg_mr(m);
	if (p) vkUnmapMemory(dev, mem);
}

int main(int argc, char **argv) {
	(void)argc; (void)argv;
	setvbuf(stdout, NULL, _IONBF, 0);
	create_instance();
	pick_device();
	create_device();
	report_features();
	drm_gate_report();

	uint64_t best = 0;
	list_modifiers(&best);

	uint32_t w = 256, h = 256, stride = 0, offset = 0;
	uint64_t mod = 0;
	int render_fd = -1;
	if (g_has_dmabuf && g_drm_node_open) {
		int dmabuf = gbm_make(0, w, h, &mod, &stride, &offset, &render_fd);
		if (dmabuf < 0) {
			fprintf(stderr, "gbm failed\n");
		} else {
			char why[64];
			int ok = modifier_supported(mod, why, sizeof why);
			printf("[import] GBM modifier 0x%016llx sampleable/importable in Vulkan: %s (%s)\n",
			       (unsigned long long)mod, ok ? OK : NO, why);
			if (ok)
				import_and_roundtrip(dmabuf, mod, stride, offset, w, h);
			else
				printf("[import] skipping import: this modifier is not usable for SAMPLED\n");
			(void)best;
		}
	} else {
		printf("\n[import] skipping dma-buf import: %s\n",
		       !g_has_dmabuf ? "VK_EXT_external_memory_dma_buf not advertised (DRM access?)"
		                     : "DRM render node not openable");
	}

	host_import_test();
	rdma_test();

	printf("\nverdict: see the OK/FAIL lines above.\n");
	return 0;
}
