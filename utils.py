import cv2
import time
import numpy as np

import torch

import open3d

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def value_tracker(vis, num, value, value_plot):
    ''' num, loss_value, are Tensor '''
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

def acc_check(net, device, test_set, test_set_loader, epoch, save_path):
    net.eval()
    net.is_training = False
    with torch.no_grad():
        total_time = 0
        for i, data in enumerate(test_set_loader, 0):
            start = time.time()
            # images, name = data
            images, labels, name = data

            images = images.to(device)

            outputs = net(images)
            process_time = time.time() - start
            print("Process Time in a Validation Image {}".format(process_time))
            total_time += process_time

            # outputs = decode_segmap(outputs[0])
            outputs = (outputs[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

            # name = test_set.color_lidar_dir[i]

            # cv2.imshow("outputs[0]", outputs[:,:,0].astype(np.uint8))
            # cv2.imshow("outputs[1]", outputs[:, :, 1].astype(np.uint8))
            # cv2.imshow("outputs[2]", outputs[:, :, 2].astype(np.uint8))
            # cv2.waitKey(0)

            # if name[0].split("\\")[-1].split(".")[0] == 'umm_000013':
            #     c0 = outputs[:, :, 0]
            #     c1 = outputs[:, :, 1]
            #     c2 = outputs[:, :, 2]
            #
            #     c0 = (255.0 * (c0 - c0.min()) / (c0.max() - c0.min())).astype('uint8')
            #     c1 = (255.0 * (c1 - c1.min()) / (c1.max() - c1.min())).astype('uint8')
            #     c2 = (255.0 * (c2 - c2.min()) / (c2.max() - c2.min())).astype('uint8')
            #
            #     cv2.imshow("outputs[0]", c0)
            #     cv2.imshow("outputs[1]", c1)
            #     cv2.imshow("outputs[2]", c2)
            #     cv2.waitKey(0)
            #     a = 0

            # c0 = outputs[:, :, 0]
            # c1 = outputs[:, :, 1]
            # c2 = outputs[:, :, 2]
            #
            # c0 = (255.0 * (c0 - c0.min()) / (c0.max() - c0.min())).astype('uint8')
            # c1 = (255.0 * (c1 - c1.min()) / (c1.max() - c1.min())).astype('uint8')
            # c2 = (255.0 * (c2 - c2.min()) / (c2.max() - c2.min())).astype('uint8')
            #
            # cv2.imshow("outputs[0]", c0)
            # cv2.imshow("outputs[1]", c1)
            # cv2.imshow("outputs[2]", c2)
            # cv2.waitKey(0)

            save_name = save_path + "{0}_{1}.png".format(epoch, name[0].split("\\")[-1].split(".")[0])
            # save_name = save_path + "{0}_{1}.png".format(epoch, name[0].split("/")[-1].split(".")[0])
            cv2.imwrite(save_name, outputs.astype(np.uint8))

        print("Average time of total process {}".format(total_time / test_set_loader.__len__()))
    net.is_training = True
    net.train()

def decode_segmap(output, nc=3):
    output = torch.argmax(output, dim=0).detach().cpu().numpy()

    r = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    b = np.zeros_like(output).astype(np.uint8)

    label_colors = np.array([(255, 0, 255), (0, 0, 255), (0, 0, 0)])  # road, non-road, background

    for l in range(0, nc):
        idx = output == l
        r[idx] = label_colors[l, 2]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 0]
    rgb = np.stack([b, g, r], axis=2)

    return rgb

def inference_check(net, device, raw_data_loader, save_path=None):
    net.eval()
    net.is_training = False
    with torch.no_grad():
        total_time = 0
        for i, data in enumerate(raw_data_loader, 0):
            start = time.time()

            images, theta_, phi_, u, v, scan_x, scan_y, scan_z, b, g, r, name = data

            print("Images.shape", images.shape)
            images = images.to(device)

            outputs = net(images)
            process_time = time.time() - start

            print("Process Time in a Validation Image {}".format(process_time))
            total_time += process_time

            outputs = (outputs[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            # cv2.imshow("imge", outputs)
            # cv2.waitKey(30)
            theta_ = theta_.detach().cpu().numpy()
            phi_ = phi_.detach().cpu().numpy()
            scan_x = scan_x.detach().cpu().numpy()
            scan_y = scan_y.detach().cpu().numpy()
            scan_z = scan_z.detach().cpu().numpy()
            b = b.detach().cpu().numpy()
            g = g.detach().cpu().numpy()
            r = r.detach().cpu().numpy()

            road_velodyne = outputs[:, :, 0][theta_, phi_]
            road_velodyne[road_velodyne >= 200] = 255
            road_velodyne[road_velodyne < 200] = 0
            non_road_velodyne = 255 - road_velodyne

            # print("road_velodyne.shape", road_velodyne.shape)
            # print("scan_x.shape", scan_x.shape)
            # print("scan_y.shape", scan_y.shape)
            # print("scan_z.shape", scan_z.shape)

            norm_non_road_velodyne = (non_road_velodyne - non_road_velodyne.min()) / (non_road_velodyne.max() - non_road_velodyne.min())

            b = (b * norm_non_road_velodyne).astype(np.uint8)
            g = ((g * norm_non_road_velodyne) + road_velodyne).astype(np.uint8)
            r = (r * norm_non_road_velodyne).astype(np.uint8)

            if name[0].split("\\")[-1].split(".")[0] == "0000000008":
                save_result_velodyne_file = open("result_velodyne.txt", "w")
                for i in range(len(b[0])):
                    temp_data = "{0} {1} {2} {3} {4} {5}\n".format(scan_x[0][i], scan_y[0][i], scan_z[0][i], r[0][i],
                                                                   g[0][i], b[0][i])
                    save_result_velodyne_file.write(temp_data)

                save_result_velodyne_file.close()

            # b = (b - b.min()) / (b.max() - b.min())
            # g = (g - g.min()) / (g.max() - g.min())
            # r = (r - r.min()) / (r.max() - r.min())
            #
            # reconstructed_velodyne_points = np.vstack((scan_x, scan_y, scan_z, r, g, b)).T
            #
            # pcd = open3d.geometry.PointCloud()
            # pcd.points = open3d.utility.Vector3dVector(reconstructed_velodyne_points[:,:3])
            # pcd.colors = open3d.utility.Vector3dVector(reconstructed_velodyne_points[:,3:])
            # open3d.visualization.draw_geometries([pcd])

            # new_img = np.zeros((384, 1242, 3), dtype=np.int32)
            new_img = np.ones((384, 1242, 3), dtype=np.int32)
            # new_img *= 48
            new_img[v, u, 0] = b
            new_img[v, u, 1] = g
            new_img[v, u, 2] = r

            kernel = np.ones((3, 3), np.uint8)
            result = cv2.dilate(new_img.astype(np.uint8), kernel, iterations=1)
            # cv2.imshow("new_img", result.astype(np.uint8))

            # save_name = save_path + "{}.png".format(name[0].split("\\")[-1].split(".")[0])
            save_name_new_img = save_path + "new_img_{}.png".format(name[0].split("\\")[-1].split(".")[0])

            # cv2.imshow("output", outputs.astype(np.uint8))
            # cv2.waitKey(30)
            # cv2.imwrite(save_name, outputs.astype(np.uint8))
            cv2.imwrite(save_name_new_img, new_img.astype(np.uint8))

        print("Average time of total process {}".format(total_time / raw_data_loader.__len__()))